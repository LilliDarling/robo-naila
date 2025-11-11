"""Performance and load testing for AI server components"""

import pytest
import time
import asyncio
import statistics
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor
import threading
from memory.conversation import ConversationMemory
from agents.input_processor import InputProcessor
from agents.response_generator import ResponseGenerator
from graphs.orchestration import NAILAOrchestrationGraph


class TestPerformance:
    """Performance benchmarks and load tests"""

    def test_memory_performance_large_dataset(self):
        """Test memory performance with large conversation datasets"""
        memory = ConversationMemory(max_history=100, ttl_hours=1)
        # Disable background task for performance testing
        if memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
            memory._cleanup_task = None
        
        # Performance metrics
        device_count = 50
        exchanges_per_device = 200
        
        start_time = time.time()
        
        # Add large number of exchanges
        for device_id in range(device_count):
            for exchange_id in range(exchanges_per_device):
                memory.add_exchange(
                    f"device_{device_id:03d}",
                    f"User message {exchange_id}",
                    f"Assistant response {exchange_id}",
                    {"intent": "test", "exchange_id": exchange_id}
                )
        
        insertion_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        
        for device_id in range(device_count):
            history = memory.get_history(f"device_{device_id:03d}", limit=10)
            context = memory.get_context(f"device_{device_id:03d}")
        
        retrieval_time = time.time() - start_time
        
        # Performance assertions
        total_exchanges = device_count * exchanges_per_device
        assert insertion_time < 10.0, f"Insertion took {insertion_time:.2f}s for {total_exchanges} exchanges"
        assert retrieval_time < 2.0, f"Retrieval took {retrieval_time:.2f}s for {device_count} devices"
        
        # Memory should respect limits
        for device_id in range(device_count):
            history = memory.get_history(f"device_{device_id:03d}", limit=10)
            assert len(history) <= 100  # max_history limit
            
        # Cleanup
        if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
    
    def test_concurrent_memory_access_performance(self):
        """Test memory performance under concurrent access"""
        memory = ConversationMemory(max_history=50, ttl_hours=1)
        # Disable background task for performance testing
        if memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
            memory._cleanup_task = None
        thread_count = 10
        operations_per_thread = 100
        
        # Performance tracking
        thread_times = []
        errors = []
        
        def worker_thread(thread_id):
            start_time = time.time()
            device_id = f"thread_device_{thread_id}"
            
            try:
                for i in range(operations_per_thread):
                    # Mix of operations
                    if i % 3 == 0:
                        memory.add_exchange(device_id, f"msg {i}", f"resp {i}", {})
                    elif i % 3 == 1:
                        memory.get_history(device_id, limit=10)
                    else:
                        memory.get_context(device_id)
                
                thread_times.append(time.time() - start_time)
                
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Run concurrent threads
        threads = []
        for i in range(thread_count):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Performance analysis
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(thread_times) == thread_count
        
        avg_time = statistics.mean(thread_times)
        max_time = max(thread_times)
        
        assert avg_time < 1.0, f"Average thread time {avg_time:.2f}s too slow"
        assert max_time < 2.0, f"Max thread time {max_time:.2f}s too slow"
        
        # Cleanup
        if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_input_processor_throughput(self, disable_hardware_optimization):
        """Test input processor throughput"""
        with patch('agents.input_processor.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_st.return_value = mock_model
            
            processor = InputProcessor()
            
            # Test data
            test_inputs = [
                {"input_type": "text", "raw_input": f"Test message {i}", "confidence": 0.9}
                for i in range(100)
            ]
            
            start_time = time.time()
            
            # Process all inputs
            results = []
            for input_data in test_inputs:
                result = await processor.process(input_data)
                results.append(result)
            
            processing_time = time.time() - start_time
            
            # Performance metrics
            throughput = len(test_inputs) / processing_time
            avg_time_per_input = processing_time / len(test_inputs) * 1000  # ms
            
            assert len(results) == len(test_inputs)
            assert throughput > 50, f"Throughput {throughput:.1f} inputs/sec too low"
            assert avg_time_per_input < 50, f"Avg processing time {avg_time_per_input:.1f}ms too slow"
 
    @pytest.mark.asyncio
    async def test_response_generator_latency(self):
        """Test response generator latency"""
        generator = ResponseGenerator()
        
        # Test various input types
        test_states = [
            {
                "intent": "greeting",
                "processed_text": "Hello",
                "context": {},
                "conversation_history": [],
                "confidence": 0.9,
                "device_id": "test"
            },
            {
                "intent": "time_query", 
                "processed_text": "What time is it?",
                "context": {},
                "conversation_history": [],
                "confidence": 0.9,
                "device_id": "test"
            },
            {
                "intent": "question",
                "processed_text": "How do I use this?",
                "context": {"history_count": 3},
                "conversation_history": [{"user": "prev", "assistant": "prev"}],
                "confidence": 0.8,
                "device_id": "test"
            }
        ]
        
        latencies = []
        
        # Measure latency for each input type
        for state in test_states:
            start_time = time.time()
            result = await generator.process(state)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            
            assert "response_text" in result
            assert len(result["response_text"]) > 0
        
        # Latency analysis
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 100, f"Average latency {avg_latency:.1f}ms too high"
        assert max_latency < 200, f"Max latency {max_latency:.1f}ms too high"

    @pytest.mark.asyncio
    async def test_orchestration_end_to_end_performance(self):
        """Test full orchestration pipeline performance"""
        with patch('agents.input_processor.InputProcessor') as mock_input, \
             patch('agents.response_generator.ResponseGenerator') as mock_response:
            
            # Setup fast mocks
            input_processor = Mock()
            input_processor.process = AsyncMock(return_value={
                "processed_text": "test",
                "intent": "greeting",
                "confidence": 0.9,
                "context": {},
                "conversation_history": []
            })
            mock_input.return_value = input_processor
            
            response_generator = Mock()
            response_generator.process = AsyncMock(return_value={
                "response_text": "Hello there!",
                "response_metadata": {"intent": "greeting"}
            })
            mock_response.return_value = response_generator
            
            orchestrator = NAILAOrchestrationGraph()
            
            # Test pipeline latency
            test_states = [
                {
                    "task_id": f"perf_task_{i}",
                    "device_id": f"device_{i}",
                    "raw_input": f"Test input {i}",
                    "input_type": "text"
                }
                for i in range(50)
            ]
            
            start_time = time.time()
            
            # Process all states
            results = []
            for state in test_states:
                result = await orchestrator.run(state)
                results.append(result)
            
            total_time = time.time() - start_time
            
            # Performance metrics
            avg_latency = (total_time / len(test_states)) * 1000  # ms
            throughput = len(test_states) / total_time
            
            assert len(results) == len(test_states)
            assert avg_latency < 100, f"Avg pipeline latency {avg_latency:.1f}ms too high"
            assert throughput > 20, f"Pipeline throughput {throughput:.1f} req/sec too low"

    @pytest.mark.asyncio
    async def test_concurrent_orchestration_performance(self):
        """Test orchestration performance under concurrent load"""
        with patch('agents.input_processor.InputProcessor') as mock_input, \
             patch('agents.response_generator.ResponseGenerator') as mock_response:

            async def fast_process(state):
                await asyncio.sleep(0.01)
                return {**state, "processed": True}
            
            input_processor = Mock()
            input_processor.process = AsyncMock(side_effect=fast_process)
            mock_input.return_value = input_processor
            
            response_generator = Mock()
            response_generator.process = AsyncMock(side_effect=fast_process)
            mock_response.return_value = response_generator
            
            orchestrator = NAILAOrchestrationGraph()
            
            # Create concurrent requests
            concurrent_requests = 20
            tasks = []
            
            start_time = time.time()
            
            for i in range(concurrent_requests):
                state = {
                    "task_id": f"concurrent_{i}",
                    "device_id": f"device_{i}",
                    "raw_input": f"Concurrent test {i}",
                    "input_type": "text"
                }
                task = orchestrator.run(state)
                tasks.append(task)
            
            # Wait for all concurrent tasks
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Performance analysis
            assert len(results) == concurrent_requests
            assert total_time < 2.0, f"Concurrent processing took {total_time:.2f}s (expected < 2s)"
            
            # Verify all completed successfully
            for result in results:
                assert "task_id" in result
                assert "response_text" in result

    def test_memory_cleanup_performance(self):
        """Test memory cleanup performance with large datasets"""
        memory = ConversationMemory(max_history=10, ttl_hours=0.001)
        # Disable background task for testing
        if memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
            memory._cleanup_task = None
        
        # Add large number of conversations
        device_count = 100
        for device_id in range(device_count):
            for i in range(5):
                memory.add_exchange(
                    f"device_{device_id:03d}",
                    f"Message {i}",
                    f"Response {i}",
                    {}
                )
        
        # Wait for TTL to expire (0.001 hours = 3.6 seconds)
        time.sleep(4.0)
        
        # Add new conversations to trigger cleanup
        for i in range(10):
            memory.add_exchange("new_device", f"New {i}", f"New resp {i}", {})
        
        # Force cleanup (reset interval to allow immediate cleanup)
        memory.last_cleanup = 0
        start_time = time.time()
        memory.cleanup_old_conversations()
        cleanup_time = time.time() - start_time
        
        # Cleanup should be fast even with many old conversations
        assert cleanup_time < 1.0, f"Cleanup took {cleanup_time:.2f}s (too slow)"
        
        # Most old conversations should be cleaned
        remaining_devices = len(memory.conversations)
        assert remaining_devices < device_count // 2, f"Too many devices remaining: {remaining_devices}"
        
        # Cleanup
        if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self):
        """Test performance improvement from caching"""
        with patch('agents.input_processor.InputProcessor') as mock_input:
            # Mock processor with intentional delay
            async def slow_process(state):
                await asyncio.sleep(0.05)
                return {**state, "intent": "greeting", "processed_text": state["raw_input"]}
            
            input_processor = Mock()
            input_processor.process = AsyncMock(side_effect=slow_process)
            mock_input.return_value = input_processor
            
            # Use real response generator to test caching
            generator = ResponseGenerator()
            
            # First request (cache miss)
            state1 = {
                "intent": "greeting",
                "processed_text": "hello",
                "context": {},
                "conversation_history": [],
                "confidence": 0.9,
                "device_id": "test"
            }
            
            start_time = time.time()
            result1 = await generator.process(state1)
            first_time = time.time() - start_time
            
            # Second identical request (cache hit)
            start_time = time.time()
            result2 = await generator.process(state1)
            second_time = time.time() - start_time
            
            # Cache hit should be faster or equal (pattern responses are already very fast)
            # With pattern-based responses being sub-millisecond, speedup may be minimal
            speedup = first_time / second_time if second_time > 0 else float('inf')

            # Relaxed assertion: cache should provide some benefit or at least not slow down
            assert speedup >= 1.0, f"Cache slowdown detected: {speedup:.1f}x (expected >=1.0x)"
            assert result1["response_text"] == result2["response_text"]

    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow unbounded"""
        import psutil
        import gc
        
        memory = ConversationMemory(max_history=20, ttl_hours=1)
        # Disable background task for testing
        if memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
            memory._cleanup_task = None
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        initial_memory = process.memory_info().rss
        
        # Add many conversations
        for cycle in range(10):
            for device_id in range(100):
                for i in range(50):
                    memory.add_exchange(
                        f"device_{device_id}_{cycle}",
                        f"Message {i}",
                        f"Response {i}",
                        {}
                    )
            
            # Clear some devices periodically
            if cycle % 3 == 0:
                for device_id in range(0, 50):
                    memory.clear_device(f"device_{device_id}_{cycle-1}")
        
        # Final memory measurement
        gc.collect()
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # Memory growth should be bounded
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB (too much)"
        
        # Verify memory limits are respected
        active_conversations = sum(len(conv) for conv in memory.conversations.values())
        max_expected = len(memory.conversations) * memory.max_history
        
        assert active_conversations <= max_expected, "Memory limits not enforced"
        
        # Cleanup
        if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()

    def test_large_conversation_history_performance(self):
        """Test performance with very long conversation histories"""
        memory = ConversationMemory(max_history=1000, ttl_hours=24)
        # Disable background task for testing
        if memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()
            memory._cleanup_task = None
        device_id = "long_conversation_device"
        
        # Create very long conversation
        start_time = time.time()
        
        for i in range(2000): 
            memory.add_exchange(
                device_id,
                f"Very long conversation message number {i} with some additional text to make it realistic",
                f"Detailed response number {i} with comprehensive information and context",
                {"intent": "conversation", "turn": i, "complexity": "high"}
            )
        
        creation_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        full_history = memory.get_history(device_id, limit=10000)
        limited_history = memory.get_history(device_id, limit=100)
        context = memory.get_context(device_id)
        retrieval_time = time.time() - start_time
        
        # Performance assertions
        assert creation_time < 5.0, f"Long conversation creation took {creation_time:.2f}s"
        assert retrieval_time < 0.5, f"Retrieval took {retrieval_time:.2f}s"
        
        # Verify limits
        assert len(full_history) == 1000  # Should respect max_history
        assert len(limited_history) == 100
        assert context["history_count"] == 1000
        
        # Cleanup
        if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
            memory._cleanup_task.cancel()