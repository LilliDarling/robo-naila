"""Test fixtures for memory components."""

import pytest
import factory

from memory.conversation import ConversationMemory


@pytest.fixture
def clean_memory():
    """Fresh in-memory conversation store. Per-test isolation."""
    return ConversationMemory(db_path=":memory:")


@pytest.fixture
def populated_memory():
    """Memory pre-seeded with two devices' worth of test exchanges."""
    memory = ConversationMemory(db_path=":memory:")
    memory.commit_exchange(
        "robot_001",
        "Hello",
        "Hi there! How can I help you?",
        intent="greeting",
        metadata={},
    )
    memory.commit_exchange(
        "robot_001",
        "What time is it?",
        "The current time is 10:30 AM",
        intent="time_query",
        metadata={},
    )
    memory.commit_exchange(
        "robot_002",
        "Weather today?",
        "I don't have weather data yet",
        intent="weather_query",
        metadata={},
    )
    return memory


@pytest.fixture
def memory_with_history():
    """Memory pre-seeded with a single device's longer conversation."""
    memory = ConversationMemory(db_path=":memory:")
    device_id = "test_device"

    exchanges = [
        ("Hi", "Hello! How can I help?", "greeting"),
        ("What's your name?", "I'm NAILA, your AI assistant", "question"),
        ("What time is it?", "The current time is 2:30 PM", "time_query"),
        ("Thank you", "You're welcome!", "gratitude"),
        ("How's the weather?", "I don't have weather access yet", "weather_query"),
        ("Can you help me?", "Of course! What do you need help with?", "question"),
        ("What can you do?", "I can answer questions and have conversations", "question"),
        ("That's great", "I'm glad you think so!", "general"),
    ]

    for user_msg, assistant_msg, intent in exchanges:
        memory.commit_exchange(device_id, user_msg, assistant_msg, intent=intent, metadata={})

    return memory


class ConversationExchangeFactory(factory.Factory):
    """Factory for the dict shape returned by ``recall_recent``."""

    class Meta:
        model = dict

    user = "Test user message"
    assistant = "Test assistant response"
    intent = "general"
    ts = factory.Sequence(lambda n: 1_700_000_000_000 + n)
    metadata = factory.Dict({})


@pytest.fixture
def sample_exchanges():
    """Sample exchange dicts in the new ``recall_recent`` shape."""
    return ConversationExchangeFactory.build_batch(3)
