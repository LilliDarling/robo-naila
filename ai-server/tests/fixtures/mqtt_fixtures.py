"""Test fixtures for MQTT components"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone
from mqtt.core.models import MQTTMessage, TopicCategory
import factory


class MQTTMessageFactory(factory.Factory):
    """Factory for creating MQTTMessage test data"""
    
    class Meta:
        model = MQTTMessage
    
    topic = "naila/ai/orchestration/main/task"
    payload = factory.Dict({
        "task_id": factory.Sequence(lambda n: f"task_{n}"),
        "device_id": factory.Sequence(lambda n: f"device_{n}"),
        "transcription": "Hello NAILA",
        "confidence": 0.95
    })
    category = TopicCategory.AI
    device_id = factory.LazyAttribute(lambda obj: obj.payload["device_id"])
    message_type = "orchestration"
    subtype = "main"
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc).isoformat())
    qos = 1
    binary_data = None


@pytest.fixture
def mock_mqtt_service():
    """Mock MQTT service for testing"""
    mock = Mock()
    mock.publish = Mock()
    mock.subscribe = Mock()
    mock.register_handler = Mock()
    mock.published_messages = []

    def capture_publish(topic, payload, qos=0):
        mock.published_messages.append({
            "topic": topic,
            "payload": payload,
            "qos": qos
        })
    
    mock.publish.side_effect = capture_publish
    return mock


@pytest.fixture
def mqtt_message():
    """Basic MQTT message for testing"""
    return MQTTMessageFactory.build()


@pytest.fixture
def stt_message():
    """STT result message"""
    return MQTTMessageFactory.build(
        topic="naila/ai/processing/stt/robot_001",
        payload={
            "task_id": "stt_task_001",
            "device_id": "robot_001", 
            "transcription": "What time is it?",
            "confidence": 0.92,
            "language": "en",
            "processing_time_ms": 450
        }
    )


@pytest.fixture
def orchestration_message():
    """Main orchestration message"""
    return MQTTMessageFactory.build(
        payload={
            "task_id": "orch_task_001",
            "device_id": "robot_001",
            "transcription": "Hello there!",
            "confidence": 0.95,
            "intent": "greeting",
            "priority": "normal"
        }
    )


@pytest.fixture
def vision_message():
    """Vision analysis message"""
    return MQTTMessageFactory.build(
        topic="naila/ai/processing/vision/robot_001",
        payload={
            "device_id": "robot_001",
            "objects_detected": [
                {
                    "class": "person",
                    "confidence": 0.96,
                    "bounding_box": {"x": 120, "y": 80, "width": 160, "height": 200}
                }
            ],
            "scene_description": "Person at desk",
            "processing_time_ms": 89
        }
    )


@pytest.fixture
def malformed_message():
    """Malformed message for error testing"""
    return MQTTMessageFactory.build(
        payload={"invalid": "data"},
        device_id=None
    )