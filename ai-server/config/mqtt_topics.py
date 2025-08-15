"""MQTT topic configuration for AI server"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputTopics:
    """Topics the AI server subscribes to"""
    stt_result = "naila/ai/processing/stt/+"
    vision_analysis = "naila/ai/processing/vision/+"
    sensor_data = "naila/device/+/sensor/+"
    main_task = "naila/ai/orchestration/main/task"
    context_update = "naila/ai/context/update"


@dataclass(frozen=True)
class OutputTopics:
    """Topics the AI server publishes to"""
    # AI responses (command server subscribes to these)
    ai_response_text = "naila/ai/responses/text"
    ai_response_audio = "naila/ai/responses/audio"
    ai_response_emotion = "naila/ai/responses/emotion"
    
    # Status and monitoring
    ai_status = "naila/ai/status"
    ai_metrics = "naila/ai/metrics"


@dataclass(frozen=True)
class CommandServerTopics:
    """Command server topics for reference"""
    # Input
    requests = "naila/command/request"
    batch_requests = "naila/command/batch"
    
    # Output to devices
    device_command = "naila/device/{device_id}/command/{command_type}"
    device_audio = "naila/device/{device_id}/audio"
    device_display = "naila/device/{device_id}/display"


# Instances
INPUT = InputTopics()
OUTPUT = OutputTopics()
COMMAND_SERVER = CommandServerTopics()