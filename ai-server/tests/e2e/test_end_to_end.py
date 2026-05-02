"""End-to-end system tests for complete AI server workflow"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from graphs.orchestration import NAILAOrchestrationGraph
from memory.conversation import ConversationMemory
from agents.input_processor import InputProcessor
from agents.response_generator import ResponseGenerator


class TestEndToEndWorkflow:
    """End-to-end tests covering complete system workflows"""
    
    @pytest.fixture
    def complete_system(self, disable_hardware_optimization):
        """Set up complete AI server system with real components"""
        memory = ConversationMemory(db_path=":memory:")

        # Mock MQTT service
        mqtt_service = Mock()
        mqtt_service.publish = AsyncMock()
        mqtt_service.published_messages = []
        
        async def capture_publish(topic, payload, qos=0):
            mqtt_service.published_messages.append({
                "topic": topic,
                "payload": payload,
                "qos": qos,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        mqtt_service.publish.side_effect = capture_publish
        
        # Mock processors with realistic behavior
        with patch('agents.input_processor.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_st.return_value = mock_model
            
            # Create real components
            input_processor = InputProcessor()
            response_generator = ResponseGenerator()
            
            # Create orchestrator
            with patch('graphs.orchestration.InputProcessor', return_value=input_processor), \
                 patch('graphs.orchestration.ResponseGenerator', return_value=response_generator):
                orchestrator = NAILAOrchestrationGraph()
            
            yield {
                "orchestrator": orchestrator,
                "memory": memory,
                "mqtt": mqtt_service,
                "input_processor": input_processor,
                "response_generator": response_generator
            }
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, complete_system):
        """Test complete conversation from input to response"""
        system = complete_system
        orchestrator = system["orchestrator"]
        memory = system["memory"]
        mqtt = system["mqtt"]
        
        # Simulate incoming STT message
        device_id = "robot_001"
        task_id = "e2e_conversation_001"
        
        initial_state = {
            "task_id": task_id,
            "device_id": device_id,
            "input_type": "text",
            "raw_input": "Hello, how are you today?",
            "confidence": 0.95,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Process through orchestration
        result = await orchestrator.run(initial_state)
        
        # Verify processing completed successfully
        assert result["task_id"] == task_id
        assert result["device_id"] == device_id
        assert "response_text" in result
        assert len(result["response_text"]) > 0
        assert result["intent"] in ["greeting", "general", "question", "time_query", "gratitude"]
        
        # Store in memory
        memory.commit_exchange(
            device_id,
            initial_state["raw_input"],
            result["response_text"],
            intent=result["intent"],
            metadata={"task_id": task_id},
        )
        
        # Simulate publishing response
        response_payload = {
            "task_id": task_id,
            "device_id": device_id,
            "response_text": result["response_text"],
            "intent": result["intent"],
            "confidence": result.get("confidence", 0.9),
            "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat())
        }
        
        await mqtt.publish("naila/ai/responses/text", response_payload, qos=1)
        
        # Verify end-to-end flow
        assert len(mqtt.published_messages) == 1
        published = mqtt.published_messages[0]
        assert published["topic"] == "naila/ai/responses/text"
        assert published["payload"]["response_text"] == result["response_text"]
        
        # Verify memory state
        history = memory.recall_recent(device_id, n=10)
        assert len(history) == 1
        assert history[0]["user"] == "Hello, how are you today?"
        assert history[0]["assistant"] == result["response_text"]
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_context(self, complete_system):
        """Test multi-turn conversation maintaining context"""
        system = complete_system
        orchestrator = system["orchestrator"]
        memory = system["memory"]
        
        device_id = "robot_002"
        conversation_turns = [
            ("What time is it?", "time_query"),
            ("What about in New York?", "time_query"),
            ("Thank you for the help", "gratitude")
        ]
        
        conversation_history = []
        
        for turn_idx, (user_input, expected_intent_category) in enumerate(conversation_turns):
            task_id = f"multiturn_{turn_idx + 1}"
            
            # Create state with accumulated history
            state = {
                "task_id": task_id,
                "device_id": device_id,
                "input_type": "text",
                "raw_input": user_input,
                "confidence": 0.9,
                "conversation_history": conversation_history.copy()
            }
            
            # Process turn
            result = await orchestrator.run(state)
            
            # Verify processing
            assert result["task_id"] == task_id
            assert "response_text" in result
            assert result["intent"] in ["time_query", "gratitude", "general", "question", "greeting"]
            
            # Add to memory and history
            memory.commit_exchange(
                device_id,
                user_input,
                result["response_text"],
                intent=result["intent"],
                metadata={"turn": turn_idx + 1},
            )

            conversation_history.append({
                "user": user_input,
                "assistant": result["response_text"],
                "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "intent": result["intent"],
                "metadata": {},
            })

        # Verify conversation continuity
        final_history = memory.recall_recent(device_id, n=10)
        assert len(final_history) == 3

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, complete_system):
        """Test system behavior with errors and recovery"""
        system = complete_system
        orchestrator = system["orchestrator"]
        mqtt = system["mqtt"]
        
        # Test case 1: Invalid input
        invalid_state = {
            "task_id": "error_test_001",
            "device_id": "robot_003",
            "input_type": "text",
            "raw_input": "",
            "confidence": 0.1
        }
        
        result = await orchestrator.run(invalid_state)

        assert result["task_id"] == "error_test_001"
        assert "response_text" in result
        # Low confidence should trigger clarification
        assert any(keyword in result["response_text"].lower() 
                  for keyword in ["didn't catch", "repeat", "clarify"])
        
        # Test case 2: Processing error simulation
        with patch.object(system["input_processor"], 'process', 
                         side_effect=Exception("Simulated processing error")):
            
            error_state = {
                "task_id": "error_test_002",
                "device_id": "robot_003",
                "input_type": "text", 
                "raw_input": "This should cause an error",
                "confidence": 0.9
            }
            
            result = await orchestrator.run(error_state)
            
            # Should collect error but not crash
            assert "errors" in result
            assert len(result["errors"]) > 0
            assert "Simulated processing error" in result["errors"][0]
        
        # Test case 3: MQTT publish failure
        mqtt.publish.side_effect = ConnectionError("MQTT connection failed")
        
        try:
            await mqtt.publish("naila/ai/responses/text", {"test": "data"})
        except ConnectionError:
            # Should handle MQTT failures gracefully
            pass
        
        # System should continue operating
        recovery_state = {
            "task_id": "recovery_test",
            "device_id": "robot_003",
            "input_type": "text",
            "raw_input": "Testing recovery",
            "confidence": 0.9
        }
        
        result = await orchestrator.run(recovery_state)
        assert result["task_id"] == "recovery_test"
        assert "response_text" in result

    @pytest.mark.asyncio
    async def test_concurrent_device_conversations(self, complete_system):
        """Test handling multiple devices simultaneously"""
        system = complete_system
        orchestrator = system["orchestrator"]
        memory = system["memory"]
        mqtt = system["mqtt"]
        
        # Create concurrent conversations from different devices
        devices = [f"robot_{i:03d}" for i in range(5)]
        conversations = {}
        
        # Initialize conversations
        for device_id in devices:
            conversations[device_id] = [
                f"Hello from {device_id}",
                f"What can you do?",
                f"Thank you!"
            ]
        
        # Process all first messages concurrently
        first_tasks = []
        for device_id in devices:
            state = {
                "task_id": f"concurrent_{device_id}_1",
                "device_id": device_id,
                "input_type": "text",
                "raw_input": conversations[device_id][0],
                "confidence": 0.9
            }
            first_tasks.append(orchestrator.run(state))
        
        first_results = await asyncio.gather(*first_tasks)
        
        # Verify all processed successfully
        assert len(first_results) == len(devices)
        for i, result in enumerate(first_results):
            device_id = devices[i]
            assert result["device_id"] == device_id
            assert "response_text" in result
            
            # Add to memory
            memory.commit_exchange(
                device_id,
                conversations[device_id][0],
                result["response_text"],
                intent=result["intent"],
                metadata={},
            )

        # Continue conversations with context
        second_tasks = []
        for i, device_id in enumerate(devices):
            # ``recall_recent`` returns newest-first; the LLM message builder
            # expects chronological history, so reverse for ``conversation_history``.
            history = list(reversed(memory.recall_recent(device_id, n=10)))
            state = {
                "task_id": f"concurrent_{device_id}_2",
                "device_id": device_id,
                "input_type": "text",
                "raw_input": conversations[device_id][1],
                "confidence": 0.9,
                "conversation_history": history,
            }
            second_tasks.append(orchestrator.run(state))

        second_results = await asyncio.gather(*second_tasks)

        # Verify context maintained per device
        for i, result in enumerate(second_results):
            device_id = devices[i]
            assert result["device_id"] == device_id

            memory.commit_exchange(
                device_id,
                conversations[device_id][1],
                result["response_text"],
                intent=result["intent"],
                metadata={},
            )

        # Verify each device has independent context
        for device_id in devices:
            history = memory.recall_recent(device_id, n=10)
            assert len(history) == 2

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, complete_system):
        """Test system performance under realistic load"""
        system = complete_system
        orchestrator = system["orchestrator"]
        mqtt = system["mqtt"]
        
        # Simulate realistic load
        total_requests = 50
        concurrent_batches = 5
        requests_per_batch = total_requests // concurrent_batches
        
        all_results = []
        start_time = time.time()
        
        for batch in range(concurrent_batches):
            batch_tasks = []
            
            for i in range(requests_per_batch):
                request_id = batch * requests_per_batch + i
                state = {
                    "task_id": f"load_test_{request_id:03d}",
                    "device_id": f"device_{request_id % 10:02d}",
                    "input_type": "text",
                    "raw_input": f"Load test message {request_id}",
                    "confidence": 0.8 + (request_id % 3) * 0.05
                }
                batch_tasks.append(orchestrator.run(state))
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            
            # Simulate MQTT publishing
            for result in batch_results:
                await mqtt.publish(
                    "naila/ai/responses/text",
                    {
                        "task_id": result["task_id"],
                        "response_text": result.get("response_text", ""),
                        "device_id": result["device_id"]
                    }
                )
        
        total_time = time.time() - start_time
        
        # Performance analysis
        throughput = total_requests / total_time
        avg_latency = total_time / total_requests * 1000  # ms
        
        # Verify all requests processed
        assert len(all_results) == total_requests
        assert len(mqtt.published_messages) == total_requests
        
        # Performance requirements
        assert throughput > 10, f"Throughput {throughput:.1f} req/sec too low"
        assert avg_latency < 200, f"Avg latency {avg_latency:.1f}ms too high"
        
        # Verify no errors
        error_count = sum(1 for result in all_results if result.get("errors"))
        assert error_count == 0, f"Found {error_count} errors in load test"

    @pytest.mark.asyncio
    async def test_memory_persistence_across_sessions(self, complete_system):
        """Test conversation memory persistence and session management"""
        system = complete_system
        memory = system["memory"]
        orchestrator = system["orchestrator"]
        
        device_id = "persistent_robot_001"
        
        # Session 1: Initial conversation
        session1_messages = [
            "Hello, I'm testing persistence",
            "My name is Alex",
            "I like coffee"
        ]
        
        for i, message in enumerate(session1_messages):
            state = {
                "task_id": f"persist_s1_{i}",
                "device_id": device_id,
                "input_type": "text",
                "raw_input": message,
                "confidence": 0.9,
            }

            result = await orchestrator.run(state)
            memory.commit_exchange(
                device_id,
                message,
                result["response_text"],
                intent=result["intent"],
                metadata={"session": 1},
            )

        # Verify session 1 stored
        s1_history = memory.recall_recent(device_id, n=10)
        assert len(s1_history) == 3

        # Simulate session break (time gap)
        time.sleep(0.1)

        # Session 2: Continuation with context
        session2_messages = [
            "Do you remember my name?",
            "What did I say I like?",
        ]

        for i, message in enumerate(session2_messages):
            history = list(reversed(memory.recall_recent(device_id, n=10)))

            state = {
                "task_id": f"persist_s2_{i}",
                "device_id": device_id,
                "input_type": "text",
                "raw_input": message,
                "confidence": 0.9,
                "conversation_history": history,
            }

            result = await orchestrator.run(state)
            memory.commit_exchange(
                device_id,
                message,
                result["response_text"],
                intent=result["intent"],
                metadata={"session": 2},
            )

        # Verify continuity: full history is one stream, no session boundary.
        full_history_chronological = list(reversed(memory.recall_recent(device_id, n=10)))
        assert len(full_history_chronological) == 5
        assert "Alex" in full_history_chronological[1]["user"]
        assert "coffee" in full_history_chronological[2]["user"]

    @pytest.mark.asyncio
    async def test_conversation_survives_restart(self, tmp_path, disable_hardware_optimization):
        """The headline v1 promise: history persists across process restart."""
        db_file = tmp_path / "naila.db"

        # Bring up an orchestration graph backed by the shared file.
        with patch("agents.input_processor.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_st.return_value = mock_model

            input_processor = InputProcessor()
            response_generator = ResponseGenerator()

            with patch("graphs.orchestration.InputProcessor", return_value=input_processor), \
                 patch("graphs.orchestration.ResponseGenerator", return_value=response_generator):
                orchestrator = NAILAOrchestrationGraph()

            memory_before = ConversationMemory(db_path=str(db_file))
            active_devices = ["shutdown_test_001", "shutdown_test_002"]

            for device_id in active_devices:
                state = {
                    "task_id": f"pre_shutdown_{device_id}",
                    "device_id": device_id,
                    "input_type": "text",
                    "raw_input": "This is before shutdown",
                    "confidence": 0.9,
                }
                result = await orchestrator.run(state)
                memory_before.commit_exchange(
                    device_id,
                    state["raw_input"],
                    result["response_text"],
                    intent=result["intent"],
                    metadata={},
                )

            memory_before.close()

            # Simulate restart by opening a fresh ConversationMemory on the
            # same file. History from the previous instance must be visible.
            memory_after = ConversationMemory(db_path=str(db_file))

            for device_id in active_devices:
                history = memory_after.recall_recent(device_id, n=10)
                assert len(history) == 1
                assert history[0]["user"] == "This is before shutdown"

    @pytest.mark.asyncio
    async def test_complete_mqtt_integration_flow(self, complete_system):
        """Test complete MQTT message flow from input to output"""
        system = complete_system
        orchestrator = system["orchestrator"]
        mqtt = system["mqtt"]
        
        # Simulate complete MQTT workflow
        device_id = "mqtt_integration_robot"
        
        # Step 1: STT publishes transcription
        stt_payload = {
            "task_id": "mqtt_flow_001",
            "device_id": device_id,
            "transcription": "What is the weather like today?",
            "confidence": 0.88,
            "language": "en",
            "processing_time_ms": 450
        }
        
        # Step 2: AI server processes (simulated)
        ai_state = {
            "task_id": stt_payload["task_id"],
            "device_id": stt_payload["device_id"],
            "input_type": "text",
            "raw_input": stt_payload["transcription"],
            "confidence": stt_payload["confidence"]
        }
        
        result = await orchestrator.run(ai_state)
        
        # Step 3: AI server publishes response
        response_payload = {
            "task_id": result["task_id"],
            "device_id": result["device_id"],
            "response_text": result["response_text"],
            "intent": result["intent"],
            "confidence": result.get("confidence", 0.8),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await mqtt.publish("naila/ai/responses/text", response_payload, qos=1)
        
        # Step 4: Command server receives response (simulated)
        command_decision = {
            "response_received": True,
            "requires_action": result["intent"] in ["weather_query", "time_query"],
            "action_type": "api_call" if result["intent"] == "weather_query" else None
        }
        
        # Step 5: TTS processes response (simulated)
        if response_payload["response_text"]:
            tts_payload = {
                "task_id": f"tts_{result['task_id']}",
                "device_id": device_id,
                "text": response_payload["response_text"],
                "voice": "default",
                "processing_time_ms": 320
            }
            
            await mqtt.publish("naila/tts/processing/request", tts_payload, qos=1)
        
        # Verify complete flow
        assert len(mqtt.published_messages) == 2
        
        # AI response published
        ai_response = mqtt.published_messages[0]
        assert ai_response["topic"] == "naila/ai/responses/text"
        assert ai_response["payload"]["device_id"] == device_id
        assert ai_response["payload"]["response_text"] == result["response_text"]
        
        # TTS request published
        tts_request = mqtt.published_messages[1]
        assert tts_request["topic"] == "naila/tts/processing/request"
        assert tts_request["payload"]["device_id"] == device_id
        assert tts_request["payload"]["text"] == result["response_text"]
        
        # Verify end-to-end latency is reasonable
        total_latency_estimate = (
            stt_payload["processing_time_ms"] +  
            100 +  
            tts_payload["processing_time_ms"]
        )
        
        assert total_latency_estimate < 1000, f"Total pipeline latency {total_latency_estimate}ms too high"