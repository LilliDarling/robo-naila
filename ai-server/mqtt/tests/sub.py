# Use for testing specific formats, debugging, without running full server

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.mqtt_config import MQTTConfig
from mqtt.client import NailaMQTTClient


def message_handling(topic, payload):
  print(f"{topic}: {payload}")

config = MQTTConfig()
client = NailaMQTTClient(config)
client.set_message_callback(message_handling)

if client.connect():
  client.subscribe("test_topic")

  try:
    print("Press CTRL+C to exit...")
    while True:
      pass
  except Exception:
    print("Caught an Exception, something went wrong...")
  finally:
    print("Disconnecting from the MQTT broker")
    client.disconnect()
else:
  print("Couldn't connect to broker")
  sys.exit(1)