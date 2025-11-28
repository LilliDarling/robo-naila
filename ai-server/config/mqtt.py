import os
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class MQTTConfig:
  broker_host: str = "localhost"
  broker_port: int = 1883
  client_id: str = "ai-server"
  username: Optional[str] = None
  password: Optional[str] = None
  keepalive: int = 60
  qos: int = 1
  clean_session: bool = True

  @classmethod
  def from_env(cls):
    kwargs = {}
    
    if broker_host := os.getenv("MQTT_BROKER_HOST"):
      kwargs["broker_host"] = broker_host
    if broker_port := os.getenv("MQTT_BROKER_PORT"):
      kwargs["broker_port"] = int(broker_port)
    if client_id := os.getenv("MQTT_CLIENT_ID"):
      kwargs["client_id"] = client_id
    if username := os.getenv("MQTT_USERNAME"):
      kwargs["username"] = username
    if password := os.getenv("MQTT_PASSWORD"):
      kwargs["password"] = password
    if keepalive := os.getenv("MQTT_KEEPALIVE"):
      kwargs["keepalive"] = int(keepalive)
    if qos := os.getenv("MQTT_QOS"):
      kwargs["qos"] = int(qos)
    if clean_session := os.getenv("MQTT_CLEAN_SESSION"):
      kwargs["clean_session"] = clean_session.lower() == "true"
    
    return cls(**kwargs)
  
  def __post_init__(self):
    if self.client_id == "ai-server":
      self.client_id = f"ai-server-{uuid.uuid4().hex[:12]}"