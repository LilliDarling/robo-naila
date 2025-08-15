"""Unit tests for ConversationMemory system"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock
from freezegun import freeze_time
from memory.conversation_memory import ConversationMemory


class TestConversationMemory:
    """Test cases for ConversationMemory"""

    def test_initialization(self):
        """Test memory initialization with custom parameters"""
        memory = ConversationMemory(max_history=10, ttl_hours=2)
        try:
            assert memory.max_history == 10
            assert memory.ttl == timedelta(hours=2)
            assert len(memory.conversations) == 0
            assert len(memory.device_metadata) == 0
        finally:
            # Cleanup background task
            if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
                memory._cleanup_task.cancel()

    def test_add_exchange(self, clean_memory):
        """Test adding conversation exchanges"""
        clean_memory.add_exchange(
            "device_001",
            "Hello",
            "Hi there!",
            {"intent": "greeting"}
        )
        
        history = clean_memory.get_history("device_001")
        assert len(history) == 1
        assert history[0]["user"] == "Hello"
        assert history[0]["assistant"] == "Hi there!"
        assert history[0]["metadata"]["intent"] == "greeting"

    def test_max_history_limit(self):
        """Test that history respects max limit"""
        memory = ConversationMemory(max_history=3, ttl_hours=1)
        try:
            # Add more than max
            for i in range(5):
                memory.add_exchange(
                    "device_001",
                    f"Message {i}",
                    f"Response {i}",
                    {}
                )
            
            history = memory.get_history("device_001")
            assert len(history) == 3
            assert history[0]["user"] == "Message 2"
            assert history[-1]["user"] == "Message 4"
        finally:
            # Cleanup background task
            if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
                memory._cleanup_task.cancel()

    def test_get_context(self, populated_memory):
        """Test context retrieval with metadata"""
        context = populated_memory.get_context("robot_001")
        
        assert context["device_id"] == "robot_001"
        assert context["history_count"] == 2
        assert context["total_exchanges"] == 2
        assert context["is_returning_user"] == True
        assert len(context["recent_exchanges"]) == 2
        assert context["last_intent"] == "time_query"

    def test_get_context_new_device(self, clean_memory):
        """Test context for unknown device"""
        context = clean_memory.get_context("unknown_device")
        
        assert context["device_id"] == "unknown_device"
        assert context["history_count"] == 0
        assert context["total_exchanges"] == 0
        assert context["is_returning_user"] == False
        assert len(context["recent_exchanges"]) == 0
        assert "last_intent" not in context

    def test_device_metadata_tracking(self, clean_memory):
        """Test device metadata is tracked correctly"""
        # First exchange
        clean_memory.add_exchange("device_001", "msg1", "resp1", {})
        metadata = clean_memory.device_metadata["device_001"]
        
        assert metadata["total_exchanges"] == 1
        assert metadata["total_sessions"] == 1
        assert "first_seen" in metadata
        assert "last_active" in metadata
        
        # Second exchange immediately
        clean_memory.add_exchange("device_001", "msg2", "resp2", {})
        metadata = clean_memory.device_metadata["device_001"]
        
        assert metadata["total_exchanges"] == 2
        assert metadata["total_sessions"] == 1

    def test_session_detection(self, clean_memory):
        """Test new session detection after gap"""
        # First exchange
        clean_memory.add_exchange("device_001", "msg1", "resp1", {})
        
        # Simulate 31 minute gap (new session threshold is 30 min)
        original_time = clean_memory.device_metadata["device_001"]["last_active"]
        clean_memory.device_metadata["device_001"]["last_active"] = original_time - 1860
        
        # Second exchange
        clean_memory.add_exchange("device_001", "msg2", "resp2", {})
        
        metadata = clean_memory.device_metadata["device_001"]
        assert metadata["total_sessions"] == 2

    def test_clear_device(self, populated_memory):
        """Test clearing device history"""
        populated_memory.clear_device("robot_001")
        
        history = populated_memory.get_history("robot_001")
        assert len(history) == 0
        assert "robot_001" not in populated_memory.conversations
        assert "robot_001" not in populated_memory.device_metadata
        
        # Other devices should remain
        history = populated_memory.get_history("robot_002")
        assert len(history) == 1

    def test_cleanup_old_conversations(self, clean_memory):
        """Test cleanup of old conversations"""
        current_time = time.time()
        
        # Add old exchange
        clean_memory.add_exchange("old_device", "old msg", "old resp", {})
        clean_memory.conversations["old_device"][0]["timestamp_unix"] = current_time - 90000
        
        # Add recent exchange
        clean_memory.add_exchange("new_device", "new msg", "new resp", {})
        
        # Force cleanup
        clean_memory.last_cleanup = 0  # Reset cleanup timer
        clean_memory.cleanup_old_conversations()
        
        # Old device should be removed
        assert "old_device" not in clean_memory.conversations
        assert "new_device" in clean_memory.conversations

    def test_thread_safety(self, clean_memory):
        """Test thread-safe operations"""
        errors = []
        
        def add_exchanges(device_id, count):
            try:
                for i in range(count):
                    clean_memory.add_exchange(
                        device_id,
                        f"msg {i}",
                        f"resp {i}",
                        {"thread": device_id}
                    )
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=add_exchanges,
                args=(f"device_{i}", 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Each device should have its exchanges (limited by max_history=5)
        for i in range(5):
            history = clean_memory.get_history(f"device_{i}")
            assert len(history) == 5

    def test_memory_stats(self, populated_memory):
        """Test memory statistics generation"""
        stats = populated_memory.get_memory_stats()
        
        assert stats["total_devices"] == 2
        assert stats["total_exchanges"] == 3
        assert stats["average_exchanges_per_device"] == 1.5
        assert "memory_cleanup_interval" in stats
        assert "background_cleanup_active" in stats

    def test_deque_performance(self):
        """Test deque maintains O(1) append performance"""
        memory = ConversationMemory(max_history=100, ttl_hours=1)
        try:
            # Add many exchanges
            start_time = time.time()
            for i in range(200):
                memory.add_exchange(
                    "perf_device",
                    f"msg {i}",
                    f"resp {i}",
                    {}
                )
            elapsed = time.time() - start_time
            
            # Should be fast even with many exchanges
            assert elapsed < 1.0
            
            # Should only keep max_history
            history = memory.get_history("perf_device")
            assert len(history) == 100
        finally:
            # Cleanup background task
            if hasattr(memory, '_cleanup_task') and memory._cleanup_task and not memory._cleanup_task.done():
                memory._cleanup_task.cancel()

    def test_get_history_with_limit(self, memory_with_history):
        """Test getting limited history"""
        full_history = memory_with_history.get_history("test_device")
        limited_history = memory_with_history.get_history("test_device", limit=3)
        
        assert len(limited_history) == 3
        assert limited_history == full_history[-3:]

    def test_cleanup_interval_enforcement(self, clean_memory):
        """Test cleanup doesn't run too frequently"""
        clean_memory.cleanup_interval = 3600
        clean_memory.last_cleanup = time.time() - 1800
        
        # Add old conversation
        clean_memory.add_exchange("device", "msg", "resp", {})
        clean_memory.conversations["device"][0]["timestamp_unix"] = 0
        
        # Try cleanup (should skip due to interval)
        clean_memory.cleanup_old_conversations()
        
        # Old conversation should still exist
        assert "device" in clean_memory.conversations

    def test_active_device_tracking(self, clean_memory):
        """Test active device tracking for optimization"""
        # Add exchanges for multiple devices
        clean_memory.add_exchange("active_1", "msg", "resp", {})
        clean_memory.add_exchange("active_2", "msg", "resp", {})
        
        assert "active_1" in clean_memory._active_devices
        assert "active_2" in clean_memory._active_devices
        
        # Clear one device
        clean_memory.clear_device("active_1")
        
        assert "active_1" not in clean_memory._active_devices
        assert "active_2" in clean_memory._active_devices

    def test_shutdown(self, clean_memory):
        """Test graceful shutdown"""
        # Create mock cleanup task
        mock_task = Mock()
        mock_task.done.return_value = False
        clean_memory._cleanup_task = mock_task
        
        # Shutdown should cancel task
        clean_memory.shutdown()
        
        # Verify task was cancelled
        mock_task.cancel.assert_called_once()
        mock_task.done.assert_called_once()

    @freeze_time("2025-01-15 10:30:00")
    def test_timestamp_consistency(self, clean_memory):
        """Test timestamp formats are consistent"""
        clean_memory.add_exchange("device", "msg", "resp", {})
        
        exchange = clean_memory.get_history("device")[0]
        
        # Should have both timestamp formats
        assert "timestamp" in exchange
        assert "timestamp_unix" in exchange
        
        # ISO timestamp should be parseable
        parsed = datetime.fromisoformat(exchange["timestamp"])
        assert parsed.year == 2025
        assert parsed.month == 1
        assert parsed.day == 15