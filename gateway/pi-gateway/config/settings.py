import os
from dataclasses import dataclass, field


@dataclass
class MQTTSettings:
    host: str = field(default_factory=lambda: os.getenv("MQTT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    username: str | None = field(default_factory=lambda: os.getenv("MQTT_USERNAME"))
    password: str | None = field(default_factory=lambda: os.getenv("MQTT_PASSWORD"))


@dataclass
class AIServerSettings:
    host: str = field(default_factory=lambda: os.getenv("AI_SERVER_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("AI_SERVER_PORT", "9999")))
    timeout: float = 10.0


@dataclass
class AudioSettings:
    sample_rate: int = 48000
    channels: int = 1
    frame_size: int = 960  # 20ms at 48kHz
    
    # VAD settings
    vad_mode: str = field(default_factory=lambda: os.getenv("AUDIO_VAD_MODE", "energy"))
    vad_threshold: int = field(default_factory=lambda: int(os.getenv("AUDIO_VAD_THRESHOLD", "500")))
    silence_frames: int = field(default_factory=lambda: int(os.getenv("AUDIO_SILENCE_FRAMES", "15")))  # 300ms
    
    # Buffer settings
    max_buffer_seconds: float = 30.0


@dataclass
class Settings:
    mqtt: MQTTSettings = field(default_factory=MQTTSettings)
    ai_server: AIServerSettings = field(default_factory=AIServerSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    
    # Topic prefix
    topic_prefix: str = "naila"
    
    # Gateway identity
    gateway_id: str = field(default_factory=lambda: os.getenv("GATEWAY_ID", "gateway-01"))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
