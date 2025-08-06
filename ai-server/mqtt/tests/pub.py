# Use for testing specific formats, debugging, without running full server

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from config.mqtt_config import MQTTConfig
from mqtt.client import NailaMQTTClient


config = MQTTConfig()
client = NailaMQTTClient(config)

if client.connect():
  time.sleep(1)
  client.publish("test_topic", "MQTT client is working", 0)

  time.sleep(1)
  client.disconnect()
else:
  print("Couldn't connect to broker")
  sys.exit(1)
