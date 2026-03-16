"""Simple MQTT test client for the AI orchestration pipeline.

Publishes a text task to the orchestration topic and listens for
the AI response. Tests the MQTT -> LangGraph -> MQTT flow.

Usage:
    uv run python scripts/test_mqtt_client.py
    uv run python scripts/test_mqtt_client.py "what can you do?"
"""

import json
import sys
import time
from pathlib import Path

# Ensure ai-server root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paho.mqtt.client as mqtt


BROKER_HOST = "localhost"
BROKER_PORT = 1883

# Topics (must match config/mqtt_topics.py)
TASK_TOPIC = "naila/ai/orchestration/main/task"
RESPONSE_TEXT_TOPIC = "naila/ai/responses/text"
RESPONSE_AUDIO_TOPIC = "naila/ai/responses/audio"


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "hello, what can you help me with?"
    device_id = "test-device"
    task_id = f"mqtt_test_{int(time.time() * 1000)}"

    got_response = {"text": False, "audio": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
            client.subscribe(RESPONSE_TEXT_TOPIC, qos=1)
            client.subscribe(RESPONSE_AUDIO_TOPIC, qos=1)
            print(f"Subscribed to response topics\n")

            # Publish the task
            task_data = {
                "task_id": task_id,
                "device_id": device_id,
                "input_type": "text",
                "transcription": query,
                "confidence": 0.95,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            client.publish(TASK_TOPIC, json.dumps(task_data), qos=1)
            print(f"Published task: \"{query}\"")
            print(f"Task ID: {task_id}")
            print(f"Waiting for response...\n")
        else:
            print(f"Connection failed with code {rc}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  [{msg.topic}] (binary payload, {len(msg.payload)} bytes)")
            return

        if msg.topic == RESPONSE_TEXT_TOPIC:
            got_response["text"] = True
            response = payload.get("response", {})
            print(f"--- Text Response ---")
            print(f"  Task ID:    {payload.get('task_id', '?')}")
            print(f"  Text:       {response.get('text', '?')}")
            print(f"  Intent:     {response.get('intent', '?')}")
            print(f"  Confidence: {response.get('confidence', '?')}")
            print()

        elif msg.topic == RESPONSE_AUDIO_TOPIC:
            got_response["audio"] = True
            audio_len = len(payload.get("audio_data", ""))
            print(f"--- Audio Response ---")
            print(f"  Format:     {payload.get('format', '?')}")
            print(f"  Sample Rate:{payload.get('sample_rate', '?')}")
            print(f"  Duration:   {payload.get('duration_ms', '?')}ms")
            print(f"  Audio data: {audio_len} chars (base64)")
            print()

        # Disconnect after getting the text response
        if got_response["text"]:
            client.disconnect()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test-mqtt-client")
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT}...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)

    # Run for up to 30 seconds
    client.loop_start()
    deadline = time.time() + 30
    try:
        while time.time() < deadline and not got_response["text"]:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()

    if not got_response["text"]:
        print("No response received within 30 seconds.")
        sys.exit(1)


if __name__ == "__main__":
    main()
