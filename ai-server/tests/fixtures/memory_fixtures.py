"""Test fixtures for memory components"""

import pytest
from unittest.mock import Mock
from memory.conversation_memory import ConversationMemory
import factory


@pytest.fixture
def clean_memory():
    """Fresh conversation memory instance"""
    memory = ConversationMemory(max_history=5, ttl_hours=1)
    # Disable background cleanup for tests
    if memory._cleanup_task:
        memory._cleanup_task.cancel()
        memory._cleanup_task = None
    return memory


@pytest.fixture
def populated_memory():
    """Memory instance with test data"""
    memory = ConversationMemory(max_history=5, ttl_hours=1)
    if memory._cleanup_task:
        memory._cleanup_task.cancel()
        memory._cleanup_task = None

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
    
    return memory


@pytest.fixture
def memory_with_history():
    """Memory with extensive conversation history"""
    memory = ConversationMemory(max_history=10, ttl_hours=1)
    if memory._cleanup_task:
        memory._cleanup_task.cancel()
        memory._cleanup_task = None
    
    device_id = "test_device"

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
    
    return memory


class ConversationExchangeFactory(factory.Factory):
    """Factory for conversation exchanges"""
    
    class Meta:
        model = dict
    
    user = "Test user message"
    assistant = "Test assistant response" 
    timestamp = factory.LazyFunction(lambda: "2025-01-15T10:30:00Z")
    metadata = factory.Dict({"intent": "general"})


@pytest.fixture
def sample_exchanges():
    """Sample conversation exchanges"""
    return ConversationExchangeFactory.build_batch(3)