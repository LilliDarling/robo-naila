"""Test fixtures for LangGraph states"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any
import factory


class StateFactory(factory.Factory):
    """Factory for creating NAILAState test data"""
    
    class Meta:
        model = dict
    
    device_id = factory.Sequence(lambda n: f"test_device_{n}")
    task_id = factory.Sequence(lambda n: f"task_{n}")
    input_type = "text"
    raw_input = "Hello, how are you?"
    processed_text = factory.LazyAttribute(lambda obj: obj.raw_input)
    intent = "greeting"
    confidence = 0.95
    context = factory.Dict({})
    response_text = None
    conversation_history = factory.List([])
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    errors = factory.List([])


@pytest.fixture
def basic_state():
    """Basic valid state for testing"""
    return StateFactory.build()


@pytest.fixture
def low_confidence_state():
    """State with low confidence for testing fallbacks"""
    return StateFactory.build(
        confidence=0.3,
        intent="general",
        raw_input="mumbled unclear text"
    )


@pytest.fixture
def conversation_state():
    """State with conversation history"""
    return StateFactory.build(
        conversation_history=[
            {
                "user": "What time is it?",
                "assistant": "The current time is 10:30 AM",
                "timestamp": "2025-01-15T10:30:00Z",
                "metadata": {"intent": "time_query"}
            }
        ],
        context={
            "history_count": 1,
            "is_returning_user": True
        }
    )


@pytest.fixture
def error_state():
    """State with errors for testing error handling"""
    return StateFactory.build(
        errors=["Processing error", "Network timeout"],
        confidence=0.0
    )


@pytest.fixture
def batch_states():
    """Multiple states for batch testing"""
    return StateFactory.build_batch(5)