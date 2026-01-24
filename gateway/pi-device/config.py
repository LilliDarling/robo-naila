import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # MQTT
    mqtt_host: str = field(default_factory=lambda: os.getenv("MQTT_HOST", "localhost"))
    mqtt_port: int = field(default_factory=lambda: int(os.getenv("MQTT_PORT", "1883")))
    mqtt_username: str | None = field(default_factory=lambda: os.getenv("MQTT_USERNAME"))
    mqtt_password: str | None = field(default_factory=lambda: os.getenv("MQTT_PASSWORD"))
    
    # Device identity
    device_id: str = field(default_factory=lambda: os.getenv("DEVICE_ID", "pi-mic-01"))
    
    # Audio
    sample_rate: int = 48000
    channels: int = 1
    frame_size: int = 960  # 20ms at 48kHz
    audio_device: int | None = field(
        default_factory=lambda: int(os.getenv("AUDIO_DEVICE")) if os.getenv("AUDIO_DEVICE") else None
    )
    
    # Topic prefix
    topic_prefix: str = "naila"
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
