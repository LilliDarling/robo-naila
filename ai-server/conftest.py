"""Global test configuration and fixtures"""

import asyncio
import pytest
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
from typing import Dict, Any, Generator
from memory.conversation import ConversationMemory
from datetime import datetime, timezone


@pytest.fixture(scope="session")
def event_loop():
    """Create session-wide event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_torch():
    """Mock torch for hardware detection tests"""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.cuda.device_count.return_value = 0
    mock.backends.mps.is_available.return_value = False
    return mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer model"""
    mock = MagicMock()
    mock.encode.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def disable_hardware_optimization(monkeypatch):
    """Disable hardware optimization for consistent tests"""
    def mock_detect_hardware():
        from config.hardware import HardwareInfo
        return HardwareInfo(
            device_type="cpu",
            device_name="Test CPU",
            optimization_level="minimal"
        )
    
    monkeypatch.setattr(
        "config.hardware.HardwareOptimizer._detect_hardware",
        mock_detect_hardware
    )


@pytest.fixture
def mock_logger():
    """Mock logger to capture log messages"""
    return Mock()


@pytest.fixture(autouse=True)
def reset_caches():
    """Clear LRU caches between tests"""
    yield
    # Clear any LRU caches
    from agents.input_processor import InputProcessor
    if hasattr(InputProcessor, '_detect_intent'):
        InputProcessor._detect_intent.cache_clear()


@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks"""
    return {
        "min_rounds": 5,
        "max_time": 2.0,
        "warmup": True,
        "warmup_iterations": 2
    }


@pytest.fixture
def basic_state():
    """Basic state for testing"""
    return {
        "task_id": "test_task_001",
        "device_id": "test_device",
        "input_type": "text",
        "raw_input": "Hello world",
        "confidence": 0.9,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def clean_memory():
    """Fresh conversation memory instance"""
    from unittest.mock import patch
    with patch('memory.conversation.ConversationMemory._start_background_cleanup'):
        memory = ConversationMemory(max_history=5, ttl_hours=1)
        # Ensure cleanup task is None for testing
        memory._cleanup_task = None
    yield memory


@pytest.fixture
def populated_memory():
    """Memory instance with test data"""
    from unittest.mock import patch
    with patch('memory.conversation.ConversationMemory._start_background_cleanup'):
        memory = ConversationMemory(max_history=5, ttl_hours=1)
        # Ensure cleanup task is None for testing
        memory._cleanup_task = None

    # Add some test conversations
    memory.add_exchange(
        "robot_001",
        "Hello",
        "Hi there! How can I help you?",
        {"intent": "greeting"}
    )
    memory.add_exchange(
        "robot_001",
        "What time is it?",
        "The current time is 10:30 AM",
        {"intent": "time_query"}
    )
    memory.add_exchange(
        "robot_002",
        "Weather today?",
        "I don't have weather data yet",
        {"intent": "weather_query"}
    )

    yield memory


@pytest.fixture
def memory_with_history():
    """Memory with extensive conversation history"""
    from unittest.mock import patch
    with patch('memory.conversation.ConversationMemory._start_background_cleanup'):
        memory = ConversationMemory(max_history=10, ttl_hours=1)
        # Ensure cleanup task is None for testing
        memory._cleanup_task = None

    device_id = "test_device"

    # Add a longer conversation
    exchanges = [
        ("Hi", "Hello! How can I help?", "greeting"),
        ("What's your name?", "I'm NAILA, your AI assistant", "question"),
        ("What time is it?", "The current time is 2:30 PM", "time_query"),
        ("Thank you", "You're welcome!", "gratitude"),
        ("How's the weather?", "I don't have weather access yet", "weather_query"),
        ("Can you help me?", "Of course! What do you need help with?", "question"),
        ("What can you do?", "I can answer questions and have conversations", "question"),
        ("That's great", "I'm glad you think so!", "general")
    ]

    for user_msg, assistant_msg, intent in exchanges:
        memory.add_exchange(device_id, user_msg, assistant_msg, {"intent": intent})

    yield memory