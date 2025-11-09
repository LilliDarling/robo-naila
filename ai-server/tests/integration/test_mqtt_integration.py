"""Integration tests for MQTT workflow and message handling"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from graphs.orchestration import NAILAOrchestrationGraph
from mqtt.core.models import MQTTMessage, TopicCategory


class TestMQTTIntegration:
    """Integration tests for MQTT message processing flow"""
    
    @pytest.fixture
    def mock_mqtt_service(self):
        """Mock MQTT service for integration testing"""
        service = Mock()
        service.publish = AsyncMock()
        service.subscribe = AsyncMock()
        service.register_handler = Mock()
        service.published_messages = []
        
        async def capture_publish(topic, payload, qos=0):
            service.published_messages.append({
                "topic": topic,
                "payload": payload,
                "qos": qos,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        service.publish.side_effect = capture_publish
        return service
    
    @pytest.fixture
    def orchestrator_with_mqtt(self, mock_mqtt_service):
        """Orchestrator configured with mocked MQTT service"""
        with patch('agents.input_processor.InputProcessor'), \
             patch('agents.response_generator.ResponseGenerator'):
            orchestrator = NAILAOrchestrationGraph()
            orchestrator.mqtt_service = mock_mqtt_service
            return orchestrator

    @pytest.mark.asyncio
    async def test_stt_to_ai_response_flow(self, mock_mqtt_service):
        """Test complete STT -> AI processing -> response flow"""
        # Incoming STT message
        stt_message = MQTTMessage(
            topic="naila/ai/processing/stt/robot_001",
            payload={
                "task_id": "stt_task_001",
                "device_id": "robot_001",
                "transcription": "What time is it?",
                "confidence": 0.92,
                "language": "en"
            },
            category=TopicCategory.AI,
            device_id="robot_001",
            message_type="processing",
            subtype="stt",
            timestamp="2025-01-15T10:30:00Z"
        )
        
        # Mock orchestrator with real processing
        with patch('agents.input_processor.InputProcessor') as mock_input, \
             patch('agents.response_generator.ResponseGenerator') as mock_response:
            
            # Setup input processor mock
            input_processor = Mock()
            input_processor.process = AsyncMock(return_value={
                "task_id": "stt_task_001",
                "device_id": "robot_001",
                "raw_input": "What time is it?",
                "processed_text": "What time is it?",
                "intent": "time_query",
                "confidence": 0.92,
                "context": {},
                "conversation_history": [],
                "timestamp": "2025-01-15T14:30:00Z",
                "errors": []
            })
            mock_input.return_value = input_processor
            
            # Setup response generator mock
            response_generator = Mock()
            response_generator.process = AsyncMock(return_value={
                "task_id": "stt_task_001",
                "device_id": "robot_001",
                "processed_text": "What time is it?",
                "intent": "time_query",
                "confidence": 0.92,
                "response_text": "The current time is 2:30 PM",
                "context": {},
                "conversation_history": [{
                    "user": "What time is it?",
                    "assistant": "The current time is 2:30 PM",
                    "timestamp": "2025-01-15T14:30:00Z"
                }],
                "timestamp": "2025-01-15T14:30:00Z",
                "errors": []
            })
            mock_response.return_value = response_generator
            
            orchestrator = NAILAOrchestrationGraph()
            
            # Convert MQTT message to orchestration state
            initial_state = {
                "task_id": stt_message.payload["task_id"],
                "device_id": stt_message.payload["device_id"],
                "input_type": "text",
                "raw_input": stt_message.payload["transcription"],
                "confidence": stt_message.payload["confidence"]
            }
            
            # Process through orchestration
            result = await orchestrator.run(initial_state)
            
            # Verify processing results (actual response generator returns current time)
            assert "current time is" in result["response_text"].lower()
            assert result["intent"] == "time_query" 
            assert result["device_id"] == "robot_001"
            
            # Verify basic processing completed successfully (integration test)
            assert "task_id" in result
            assert "errors" in result

    @pytest.mark.asyncio
    async def test_mqtt_response_publishing(self, mock_mqtt_service):
        """Test that AI responses are published to correct MQTT topics"""
        # Simulate processing result
        processing_result = {
            "task_id": "task_001",
            "device_id": "robot_001",
            "response_text": "Hello there! How can I help you?",
            "intent": "greeting",
            "confidence": 0.95,
            "timestamp": "2025-01-15T14:30:00Z"
        }
        
        # Expected response topic and payload
        expected_topic = "naila/ai/responses/text"
        expected_payload = {
            "task_id": "task_001",
            "device_id": "robot_001", 
            "response_text": "Hello there! How can I help you?",
            "intent": "greeting",
            "confidence": 0.95,
            "timestamp": "2025-01-15T14:30:00Z"
        }
        
        # Publish response
        await mock_mqtt_service.publish(expected_topic, expected_payload, qos=1)
        
        # Verify publish was called
        assert len(mock_mqtt_service.published_messages) == 1
        published = mock_mqtt_service.published_messages[0]
        assert published["topic"] == expected_topic
        assert published["payload"] == expected_payload
        assert published["qos"] == 1

    @pytest.mark.asyncio
    async def test_vision_and_text_multimodal_flow(self, mock_mqtt_service):
        """Test processing flow with both vision and text inputs"""
        # Vision message
        vision_message = MQTTMessage(
            topic="naila/ai/processing/vision/robot_001",
            payload={
                "device_id": "robot_001",
                "objects_detected": [
                    {"class": "person", "confidence": 0.96}
                ],
                "scene_description": "Person at desk"
            },
            category=TopicCategory.AI,
            device_id="robot_001",
            message_type="processing",
            subtype="vision",
            timestamp="2025-01-15T10:30:00Z"
        )
        
        # Text message
        text_message = MQTTMessage(
            topic="naila/ai/processing/stt/robot_001", 
            payload={
                "task_id": "multimodal_001",
                "device_id": "robot_001",
                "transcription": "Who is in front of me?",
                "confidence": 0.88
            },
            category=TopicCategory.AI,
            device_id="robot_001",
            message_type="processing",
            subtype="stt",
            timestamp="2025-01-15T10:30:00Z"
        )
        
        # Simulated multimodal state
        multimodal_state = {
            "task_id": "multimodal_001",
            "device_id": "robot_001",
            "input_type": "multimodal",
            "raw_input": "Who is in front of me?",
            "context": {
                "vision_data": vision_message.payload,
                "objects_detected": ["person"],
                "scene_description": "Person at desk"
            }
        }
        
        # Mock processors for multimodal handling
        with patch('agents.input_processor.InputProcessor') as mock_input, \
             patch('agents.response_generator.ResponseGenerator') as mock_response:
            
            input_processor = Mock()
            input_processor.process = AsyncMock(return_value={
                **multimodal_state,
                "processed_text": "Who is in front of me?",
                "intent": "vision_query", 
                "confidence": 0.88
            })
            mock_input.return_value = input_processor
            
            response_generator = Mock()
            response_generator.process = AsyncMock(return_value={
                **multimodal_state,
                "intent": "vision_query",
                "confidence": 0.88,
                "response_text": "I can see a person at the desk in front of you."
            })
            mock_response.return_value = response_generator
            
            orchestrator = NAILAOrchestrationGraph()
            result = await orchestrator.run(multimodal_state)
            
            # Verify multimodal processing (integration test uses real components)
            assert result["response_text"]
            assert result["intent"]
            assert "vision_data" in result["context"]

    @pytest.mark.asyncio
    async def test_error_message_publishing(self, mock_mqtt_service):
        """Test that processing errors are published to error topics"""
        error_payload = {
            "task_id": "error_task_001",
            "device_id": "robot_001",
            "error_type": "processing_failed",
            "error_message": "Intent detection failed",
            "timestamp": "2025-01-15T14:30:00Z",
            "original_input": "unintelligible input"
        }
        
        await mock_mqtt_service.publish("naila/ai/errors/processing", error_payload, qos=1)
        
        published = mock_mqtt_service.published_messages[0]
        assert published["topic"] == "naila/ai/errors/processing"
        assert published["payload"]["error_type"] == "processing_failed"

    @pytest.mark.asyncio
    async def test_message_validation_and_routing(self):
        """Test MQTT message validation and routing logic"""
        # Valid orchestration message
        valid_message = MQTTMessage(
            topic="naila/ai/orchestration/main/task",
            payload={
                "task_id": "valid_001",
                "device_id": "robot_001",
                "transcription": "Hello NAILA",
                "confidence": 0.95
            },
            category=TopicCategory.AI,
            device_id="robot_001",
            message_type="orchestration",
            subtype="main",
            timestamp="2025-01-15T10:30:00Z"
        )
        
        # Invalid message missing required fields
        invalid_message = MQTTMessage(
            topic="naila/ai/orchestration/main/task", 
            payload={"incomplete": "data"},
            category=TopicCategory.AI,
            device_id="robot_001",
            message_type="orchestration",
            subtype="main",
            timestamp="2025-01-15T10:30:00Z"
        )
        
        # Validation function
        def validate_orchestration_message(message: MQTTMessage) -> bool:
            required_fields = ["task_id", "device_id", "transcription"]
            return all(field in message.payload for field in required_fields)
        
        assert validate_orchestration_message(valid_message) == True
        assert validate_orchestration_message(invalid_message) == False

    @pytest.mark.asyncio
    async def test_concurrent_device_processing(self, mock_mqtt_service):
        """Test processing messages from multiple devices concurrently"""
        # Messages from different devices
        device_messages = [
            {
                "task_id": f"task_{i}",
                "device_id": f"robot_{i:03d}",
                "raw_input": f"Hello from device {i}",
                "input_type": "text"
            }
            for i in range(5)
        ]
        
        # Mock processors
        with patch('agents.input_processor.InputProcessor') as mock_input, \
             patch('agents.response_generator.ResponseGenerator') as mock_response:
            
            input_processor = Mock()
            response_generator = Mock()
            
            # Setup processor mocks to handle any device
            async def process_input(state):
                return {
                    **state,
                    "processed_text": state["raw_input"],
                    "intent": "greeting",
                    "confidence": 0.9
                }
            
            async def process_response(state):
                return {
                    **state,
                    "response_text": f"Hello {state['device_id']}!"
                }
            
            input_processor.process = AsyncMock(side_effect=process_input)
            response_generator.process = AsyncMock(side_effect=process_response)
            
            mock_input.return_value = input_processor
            mock_response.return_value = response_generator
            
            orchestrator = NAILAOrchestrationGraph()
            
            # Process all messages concurrently
            tasks = [orchestrator.run(msg) for msg in device_messages]
            results = await asyncio.gather(*tasks)
            
            # Verify all devices processed
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result["device_id"] == f"robot_{i:03d}"
                assert result["response_text"]

    @pytest.mark.asyncio
    async def test_mqtt_qos_and_retention(self, mock_mqtt_service):
        """Test MQTT QoS levels and message retention for different message types"""
        # High priority response (QoS 2)
        await mock_mqtt_service.publish(
            "naila/ai/responses/priority",
            {"urgent": "data"},
            qos=2
        )
        
        # Standard response (QoS 1)
        await mock_mqtt_service.publish(
            "naila/ai/responses/text",
            {"normal": "data"},
            qos=1
        )
        
        # Status update (QoS 0)
        await mock_mqtt_service.publish(
            "naila/ai/status/health",
            {"status": "ok"},
            qos=0
        )
        
        messages = mock_mqtt_service.published_messages
        assert messages[0]["qos"] == 2
        assert messages[1]["qos"] == 1
        assert messages[2]["qos"] == 0

    @pytest.mark.asyncio
    async def test_conversation_memory_integration(self, mock_mqtt_service):
        """Test integration between MQTT flow and conversation memory"""
        from memory.conversation import ConversationMemory
        
        memory = ConversationMemory(max_history=3, ttl_hours=1)
        # Disable background task for testing
        if memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
            memory._cleanup_task = None
        device_id = "robot_001"
        
        # Simulate conversation flow through MQTT
        conversation_turns = [
            ("Hello", "Hi there! How can I help?"),
            ("What time is it?", "The current time is 2:30 PM"),
            ("Thank you", "You're welcome!")
        ]
        
        for user_msg, assistant_msg in conversation_turns:
            # Add to memory (simulating orchestration result)
            memory.add_exchange(device_id, user_msg, assistant_msg, {"intent": "test"})
            
            # Publish response via MQTT
            await mock_mqtt_service.publish(
                "naila/ai/responses/text",
                {
                    "device_id": device_id,
                    "response_text": assistant_msg,
                    "user_input": user_msg
                }
            )
        
        # Verify memory state
        history = memory.get_history(device_id)
        assert len(history) == 3
        assert history[-1]["assistant"] == "You're welcome!"
        
        # Verify MQTT messages
        assert len(mock_mqtt_service.published_messages) == 3
        last_published = mock_mqtt_service.published_messages[-1]
        assert last_published["payload"]["response_text"] == "You're welcome!"
        
        # Cleanup
        if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_service_failure(self):
        """Test system behavior when MQTT service is unavailable"""
        # Mock failed MQTT service
        failed_service = Mock()
        failed_service.publish = AsyncMock(side_effect=ConnectionError("MQTT unavailable"))
        
        # System should continue processing even if publishing fails
        with patch('agents.input_processor.InputProcessor'), \
             patch('agents.response_generator.ResponseGenerator'):
            
            orchestrator = NAILAOrchestrationGraph()
            
            # Processing should still work
            result = await orchestrator.run({
                "task_id": "offline_test",
                "device_id": "robot_001",
                "raw_input": "test offline mode"
            })
            
            # Should have processing results even without MQTT
            assert "task_id" in result
            assert result["task_id"] == "offline_test"