# NAILA Robot MQTT Protocol Specification

## Overview

This document defines the complete MQTT communication protocol for the NAILA robot ecosystem. Think of MQTT topics like a postal system - each message has a specific address (topic) that tells the system exactly where it should go and what it contains.

## Topic Structure Hierarchy

The topic structure follows a logical hierarchy, like organizing files in folders on your computer:

```
naila/
├── devices/
│   └── <device_id>/
│       ├── sensors/          # Raw sensor data
│       ├── audio/           # Audio stream and processing
│       ├── vision/          # Camera and visual data
│       ├── status/          # Device health and state
│       └── actions/         # Motor movements and responses
├── ai/
│   ├── processing/
│   │   ├── stt/            # Speech-to-text results
│   │   ├── vision/         # Computer vision analysis
│   │   ├── emotion/        # Emotion detection results
│   │   └── agents/         # Inter-agent communication
│   └── orchestration/
│       ├── main/           # Main agent coordination
│       ├── personality/    # Personality management
│       ├── context/        # Context tracking
│       ├── memory/         # Memory operations
│       ├── planning/       # Task planning
│       └── tooling/        # External tool integration
├── command/
│   ├── request             # Command requests from AI server
│   ├── batch               # Batch command requests
│   ├── status              # Command execution status
│   └── result              # Command execution results
└── system/
    ├── health/             # System monitoring
    ├── updates/            # OTA and configuration
    ├── security/           # Security events
    └── analytics/          # Performance metrics
```

## Device Communication Topics

### Device → Server (Robot Publishing)

#### Sensor Data
```
Topic: naila/devices/{device_id}/sensors/{sensor_type}
Payload: JSON
QoS: 0 (fire and forget)
Retain: false
```

**Sensor Types:**
- `touch` - Touch sensor activation
- `proximity` - IR proximity detection
- `motion` - IMU/accelerometer data
- `environment` - Temperature, humidity, light levels
- `battery` - Power status and charging state

**Example Payloads:**
```json
// Touch sensor
{
  "timestamp": "2025-01-15T10:30:45Z",
  "sensor_id": "touch_01",
  "state": "pressed",
  "duration": 250,
  "location": {"x": 12, "y": 8}
}

// Battery status
{
  "timestamp": "2025-01-15T10:30:45Z",
  "voltage": 3.7,
  "percentage": 85,
  "charging": false,
  "temperature": 25.3
}
```

#### Audio Stream
```
Topic: naila/devices/{device_id}/audio/stream
Payload: Binary (audio chunks)
QoS: 1 (at least once)
Retain: false
```

```
Topic: naila/devices/{device_id}/audio/wake_word
Payload: JSON
QoS: 1
Retain: false
```

**Wake Word Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "detected": true,
  "confidence": 0.89,
  "wake_word": "naila",
  "audio_length_ms": 1500
}
```

#### Vision Data
```
Topic: naila/devices/{device_id}/vision/frame
Payload: Binary (JPEG image)
QoS: 1
Retain: false
```

```
Topic: naila/devices/{device_id}/vision/event
Payload: JSON
QoS: 1
Retain: false
```

**Vision Event Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "event_type": "face_detected",
  "confidence": 0.92,
  "bounding_box": {"x": 120, "y": 80, "width": 160, "height": 200},
  "person_id": "user_001",
  "emotion": "happy"
}
```

#### Device Status
```
Topic: naila/devices/{device_id}/status/heartbeat
Payload: JSON
QoS: 1
Retain: true (last will)
```

**Heartbeat Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "device_id": "naila_robot_001",
  "status": "online",
  "uptime_seconds": 3600,
  "free_memory": 245760,
  "wifi_rssi": -45,
  "firmware_version": "v1.2.3",
  "last_command_id": "cmd_12345"
}
```

### Server → Device (AI Server Publishing)

#### Audio Responses
```
Topic: naila/devices/{device_id}/actions/audio/play
Payload: Binary (TTS audio)
QoS: 1
Retain: false
```

```
Topic: naila/devices/{device_id}/actions/audio/control
Payload: JSON
QoS: 1
Retain: false
```

**Audio Control Payload:**
```json
{
  "command_id": "cmd_12346",
  "timestamp": "2025-01-15T10:30:50Z",
  "action": "play",
  "volume": 75,
  "interrupt_current": true,
  "audio_format": "wav",
  "sample_rate": 16000
}
```

#### Motor Commands
```
Topic: naila/devices/{device_id}/actions/motors/{motor_group}
Payload: JSON
QoS: 1
Retain: false
```

**Motor Groups:** `head`, `body`, `arms`, `eyes`

**Motor Command Payload:**
```json
{
  "command_id": "cmd_12347",
  "timestamp": "2025-01-15T10:30:50Z",
  "movements": [
    {
      "motor_id": "head_pan",
      "target_angle": 45,
      "speed": 50,
      "easing": "smooth"
    },
    {
      "motor_id": "head_tilt",
      "target_angle": -15,
      "speed": 30,
      "easing": "linear"
    }
  ],
  "sequence": true,
  "loop": false
}
```

#### Display Commands
```
Topic: naila/devices/{device_id}/actions/display/expression
Payload: JSON
QoS: 1
Retain: false
```

**Display Expression Payload:**
```json
{
  "command_id": "cmd_12348",
  "timestamp": "2025-01-15T10:30:50Z",
  "expression": "happy",
  "intensity": 0.8,
  "duration_ms": 2000,
  "transition": "fade",
  "eye_animation": {
    "blink_rate": 0.5,
    "pupil_size": 0.7,
    "color": "#4A90E2"
  }
}
```

## AI Agent Communication Topics

### Processing Results
```
Topic: naila/ai/processing/stt/{device_id}
Payload: JSON
QoS: 1
Retain: false
```

**STT Result Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:48Z",
  "device_id": "naila_robot_001",
  "transcription": "what time is it",
  "confidence": 0.94,
  "language": "en",
  "processing_time_ms": 450,
  "audio_duration_ms": 1200
}
```

```
Topic: naila/ai/processing/vision/{device_id}
Payload: JSON
QoS: 1
Retain: false
```

**Vision Analysis Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:48Z",
  "device_id": "naila_robot_001",
  "objects_detected": [
    {
      "class": "person",
      "confidence": 0.96,
      "bounding_box": {"x": 120, "y": 80, "width": 160, "height": 200}
    },
    {
      "class": "laptop",
      "confidence": 0.87,
      "bounding_box": {"x": 300, "y": 200, "width": 200, "height": 150}
    }
  ],
  "scene_description": "Person working at desk with laptop",
  "processing_time_ms": 89
}
```

### Agent Orchestration
```
Topic: naila/ai/orchestration/main/task
Payload: JSON
QoS: 1
Retain: false
```

**Task Coordination Payload:**
```json
{
  "task_id": "task_567",
  "timestamp": "2025-01-15T10:30:49Z",
  "device_id": "naila_robot_001",
  "user_id": "user_001",
  "intent": "time_query",
  "entities": {},
  "context": {
    "conversation_id": "conv_123",
    "turn_number": 3,
    "mood": "neutral",
    "last_interaction": "2025-01-15T10:25:30Z"
  },
  "priority": "normal"
}
```

```
Topic: naila/ai/orchestration/personality/response
Payload: JSON
QoS: 1
Retain: false
```

**Personality Response Payload:**
```json
{
  "task_id": "task_567",
  "timestamp": "2025-01-15T10:30:49Z",
  "personality_traits": {
    "friendliness": 0.9,
    "enthusiasm": 0.7,
    "formality": 0.3
  },
  "response_style": "casual_friendly",
  "emotional_tone": "helpful",
  "suggested_actions": [
    {
      "type": "expression",
      "value": "attentive"
    },
    {
      "type": "gesture",
      "value": "slight_nod"
    }
  ]
}
```

## Command Server Topics

The Command Server acts as a centralized command dispatcher to prevent race conditions and ensure proper command sequencing to devices.

### AI Server → Command Server

#### Command Request
```
Topic: naila/command/request
Payload: JSON
QoS: 1
Retain: false
```

**Command Request Payload:**
```json
{
  "task_id": "task_1234567890",
  "source": "ai_server",
  "target_device": "naila_robot_001",
  "timestamp": "2025-01-15T10:30:50Z",
  "priority": "normal",
  "response": {
    "text": "The current time is 10:30 AM",
    "intent": "time_query",
    "confidence": 0.95
  },
  "commands": [
    {
      "type": "tts",
      "sequence": 1,
      "payload": {
        "text": "The current time is 10:30 AM",
        "format": "wav",
        "sample_rate": 16000,
        "voice": "default"
      }
    },
    {
      "type": "led",
      "sequence": 2,
      "payload": {
        "pattern": "pulse",
        "color": "blue",
        "duration_ms": 2000
      }
    },
    {
      "type": "motor",
      "sequence": 3,
      "payload": {
        "motor_group": "head",
        "action": "nod",
        "speed": 40
      }
    }
  ]
}
```

#### Batch Commands
```
Topic: naila/command/batch
Payload: JSON
QoS: 1
Retain: false
```

**Batch Command Payload:**
```json
{
  "batch_id": "batch_001",
  "timestamp": "2025-01-15T10:30:50Z",
  "commands": [
    {
      "target_device": "naila_robot_001",
      "command": { /* command structure */ }
    },
    {
      "target_device": "security_drone_001",
      "command": { /* command structure */ }
    }
  ]
}
```

### Command Server → Devices

The Command Server translates high-level commands into device-specific MQTT messages:

- TTS commands → `naila/devices/{device_id}/actions/audio/play`
- Motor commands → `naila/devices/{device_id}/actions/motors/{group}`
- LED commands → `naila/devices/{device_id}/actions/display/led`
- Display commands → `naila/devices/{device_id}/actions/display/expression`

### Command Server → AI Server

#### Command Status
```
Topic: naila/command/status
Payload: JSON
QoS: 1
Retain: false
```

**Command Status Payload:**
```json
{
  "task_id": "task_1234567890",
  "device_id": "naila_robot_001",
  "timestamp": "2025-01-15T10:30:51Z",
  "status": "executing",
  "current_command": 2,
  "total_commands": 3,
  "progress": 0.67
}
```

#### Command Result
```
Topic: naila/command/result
Payload: JSON
QoS: 1
Retain: false
```

**Command Result Payload:**
```json
{
  "task_id": "task_1234567890",
  "device_id": "naila_robot_001",
  "timestamp": "2025-01-15T10:30:55Z",
  "status": "completed",
  "execution_time_ms": 4500,
  "results": [
    {
      "command_sequence": 1,
      "type": "tts",
      "status": "success",
      "duration_ms": 2100
    },
    {
      "command_sequence": 2,
      "type": "led",
      "status": "success",
      "duration_ms": 2000
    },
    {
      "command_sequence": 3,
      "type": "motor",
      "status": "success",
      "duration_ms": 400
    }
  ],
  "errors": []
}
```

## System Management Topics

### Health Monitoring
```
Topic: naila/system/health/services
Payload: JSON
QoS: 1
Retain: true
```

**Service Health Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:50Z",
  "services": {
    "stt_service": {
      "status": "healthy",
      "response_time_ms": 245,
      "error_rate": 0.01,
      "memory_usage_mb": 512
    },
    "llm_service": {
      "status": "healthy", 
      "response_time_ms": 1200,
      "error_rate": 0.005,
      "memory_usage_mb": 3072
    },
    "tts_service": {
      "status": "healthy",
      "response_time_ms": 800,
      "error_rate": 0.02,
      "memory_usage_mb": 256
    }
  },
  "system_load": 0.45,
  "total_memory_gb": 16.0,
  "free_memory_gb": 8.2
}
```

### OTA Updates
```
Topic: naila/system/updates/available/{device_id}
Payload: JSON
QoS: 1
Retain: true
```

**Update Notification Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:50Z",
  "device_id": "naila_robot_001",
  "update_available": true,
  "current_version": "v1.2.3",
  "new_version": "v1.3.0",
  "update_size_bytes": 1048576,
  "download_url": "http://192.168.1.100:8071/firmware/naila_robot_v1.3.0.bin",
  "checksum": "sha256:abc123def456...",
  "critical": false,
  "release_notes": "Bug fixes and improved wake word detection"
}
```

### Security Events
```
Topic: naila/system/security/alert
Payload: JSON
QoS: 2 (exactly once)
Retain: false
```

**Security Alert Payload:**
```json
{
  "timestamp": "2025-01-15T10:30:50Z",
  "alert_id": "sec_001",
  "severity": "medium",
  "event_type": "unauthorized_access_attempt",
  "source_ip": "192.168.1.45",
  "device_id": "naila_robot_001",
  "description": "Multiple failed authentication attempts",
  "action_taken": "temporary_ip_block"
}
```

## Message Flow Examples

### Typical Interaction Flow

1. **Wake Word Detection:**
   ```
   naila/devices/naila_robot_001/audio/wake_word
   → Contains: {"detected": true, "confidence": 0.89}
   ```

2. **Audio Streaming:**
   ```
   naila/devices/naila_robot_001/audio/stream
   → Contains: Binary audio chunks
   ```

3. **Speech Recognition:**
   ```
   naila/ai/processing/stt/naila_robot_001
   → Contains: {"transcription": "what time is it", "confidence": 0.94}
   ```

4. **Agent Processing:**
   ```
   naila/ai/orchestration/main/task
   → Contains: Task details and context
   
   naila/ai/orchestration/personality/response
   → Contains: Personality-adjusted response style
   ```

5. **Response Generation:**
   ```
   naila/ai/responses/audio/naila_robot_001
   → Contains: Generated TTS audio data
   
   naila/ai/responses/actions/naila_robot_001
   → Contains: Motor and display commands
   ```

6. **Device Execution:**
   ```
   naila/devices/naila_robot_001/actions/audio/play
   → Binary TTS audio
   
   naila/devices/naila_robot_001/actions/display/expression
   → Display expression commands
   ```

## QoS Levels Guide

Think of QoS like different shipping methods:

- **QoS 0 (At most once):** Like regular mail - fast but might get lost
  - Use for: Frequent sensor data, heartbeats
  
- **QoS 1 (At least once):** Like certified mail - guaranteed delivery, might get duplicates
  - Use for: Commands, responses, important events
  
- **QoS 2 (Exactly once):** Like registered mail - guaranteed single delivery
  - Use for: Critical security alerts, important system commands

## Retained Messages

Retained messages are like leaving a note on the fridge - the last message stays there for anyone who looks later.

**Use retained messages for:**
- Device status/heartbeat (so new connections know device state)
- System configuration
- Available updates
- Service health status

**Don't use retained messages for:**
- Audio streams
- Temporary commands
- Conversation data

## Error Handling

### Error Response Format
```json
{
  "error": true,
  "error_code": "PROCESSING_FAILED",
  "error_message": "STT service temporarily unavailable",
  "timestamp": "2025-01-15T10:30:50Z",
  "retry_after_seconds": 30,
  "correlation_id": "req_12345"
}
```

### Error Topics
```
Topic: naila/system/errors/{service_name}
Payload: JSON (error details)
QoS: 1
Retain: false
```

## Implementation Notes

### Device ID Format
- Use descriptive, unique identifiers: `naila_robot_001`, `security_drone_01`, `cat_companion_02`
- Include device type and instance number for clarity

### Timestamp Format
- Always use ISO 8601 format: `2025-01-15T10:30:45Z`
- Include timezone (Z for UTC) for consistency

### Binary Data Handling
- Audio: Raw PCM or compressed formats (specify in metadata)
- Images: JPEG for efficiency, include compression level in filename/metadata
- Use appropriate chunk sizes (4KB-8KB for audio streams)

### Performance Considerations
- Keep JSON payloads under 1MB when possible
- Use binary formats for large data (audio, images)
- Implement message compression for large text payloads
- Consider message batching for high-frequency sensor data
