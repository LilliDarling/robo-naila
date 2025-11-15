from typing import Dict, Callable, Optional
import paho.mqtt.client as mqtt
from config.mqtt import MQTTConfig
from utils import get_logger

class NailaMQTTClient:
  def __init__(self, config: MQTTConfig):
    self.config = config
    self.client: Optional[mqtt.Client] = None
    self.connected = False
    self.message_handlers: Dict[str, Callable] = {}
    self.reconnect_attempts = 0
    self.max_reconnect_attempts = 5
    self.on_message_callback: Optional[Callable] = None
    self.subscriptions = []
    self.logger = get_logger(__name__)
  
  def setup(self):
    """Initialize the MQTT Client"""
    self.client = mqtt.Client(
      client_id=self.config.client_id,
      clean_session=self.config.clean_session
    )

    if self.config.username and self.config.password:
      self.client.username_pw_set(self.config.username, self.config.password)
    
    self.client.on_connect = self._on_connect
    self.client.on_disconnect = self._on_disconnect
    self.client.on_message = self._on_message
    self.client.on_subscribe = self._on_subscribe
    self.client.on_publish = self._on_publish
  
  def _on_connect(self, client, userdata, flags, rc):
    """Callback for when the client connects"""
    if rc == 0:
      self.connected = True
      self.logger.info("mqtt_connected", broker_host=self.config.broker_host, broker_port=self.config.broker_port)

      for topic, qos in self.subscriptions:
        result = self.client.subscribe(topic, qos)
        self.logger.debug("subscribed", topic=topic, qos=qos)
    else:
      self.logger.error("connection_failed", return_code=rc)
  
  def _on_disconnect(self, client, userdata, rc):
    """Called when client disconnects"""
    self.connected = False
    if rc != 0:
      self.logger.warning("unexpected_disconnection", return_code=rc)
    else:
      self.logger.info("mqtt_disconnected")
  
  def _on_message(self, client, userdata, msg):
    """Called when message is received"""
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    self.logger.debug("message_received", topic=topic, payload=payload)

    if self.on_message_callback:
      self.on_message_callback(topic, payload)
  
  def _on_subscribe(self, client, userdata, mid, granted_qos):
    """Called when subscription is acknowledged"""
    self.logger.debug("subscription_acknowledged", message_id=mid)
  
  def _on_publish(self, client, userdata, mid):
    """Called when publish is complete"""
    self.logger.debug("message_published", message_id=mid)

  def connect(self):
    """Connect to the MQTT broker"""
    if not self.client:
      self.setup()

    try:
      result = self.client.connect(self.config.broker_host, self.config.broker_port, self.config.keepalive)
      if result == 0:
        self.client.loop_start()
        return True
      else:
        self.logger.error("connection_failed_with_result", result=result)
        return False
    except Exception as e:
      self.logger.error("connection_exception", error=str(e), error_type=type(e).__name__)
      return False

  def disconnect(self):
    """Disconnect from broker"""
    if self.client and self.connected:
      self.client.loop_stop()
      self.client.disconnect()
  
  def subscribe(self, topic, qos=0):
    """Subscribe to a topic"""
    self.subscriptions.append((topic, qos))

    if self.connected:
      result = self.client.subscribe(topic, qos)
      self.logger.debug("subscribed_to_topic", topic=topic)
      return result
    else:
      self.logger.debug("subscription_deferred", topic=topic, reason="not_connected")
      return True
  
  def publish(self, topic, message, qos=0):
    """Publish a message"""
    if self.connected:
      result = self.client.publish(topic, message, qos)
      self.logger.debug("published_message", topic=topic, message=message)
      return result
    else:
      self.logger.warning("publish_failed", reason="not_connected")
      return False
  
  def set_message_callback(self, callback: Callable):
    """Set custom message handler"""
    self.on_message_callback = callback