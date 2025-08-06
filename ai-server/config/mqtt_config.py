import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MQTTConfig:
  broker_host: str = "localhost"
  broker_port: int = 1883
  client_id: str = None
  username: Optional[str] = None
  password: Optional[str] = None
  keepalive: int = 60
  qos: int = 1
  clean_session: bool = True

  @classmethod
  def from_env(cls):
    return cls(
      broker_host=os.getenv("MQTT_BROKER_HOST", "localhost"),
      broker_port=int(os.getenv("MQTT_BROKER_PORT", "1883")),
      client_id=os.getenv("MQTT_CLIENT_ID", "ai-server"),
      username=os.getenv("MQTT_USERNAME"),
      password=os.getenv("MQTT_PASSWORD"),
      keepalive=int(os.getenv("MQTT_KEEPALIVE", "60")),
      qos=int(os.getenv("MQTT_QOS", "1")),
      clean_session=os.getenv("MQTT_CLEAN_SESSION", "true").lower() == "true"
    )
  
  def __post_init__(self):
    if self.client_id is None:
      import random
      self.client_id = f"ai-server-{random.randint(1000,9999)}"