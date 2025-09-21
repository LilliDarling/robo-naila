#include "naila_mqtt.h"
#include "esp_log.h"
#include "mqtt_client.h"
#include <stdio.h>
#include <string.h>

// MQTT Configuration - TODO: Move to Kconfig
#define CONFIG_MQTT_BROKER_IP "10.0.0.117"
#define CONFIG_MQTT_BROKER_PORT 1883
#define CONFIG_ROBOT_ID "naila_robot_001"

static const char* TAG = "NAILA_MQTT";
static esp_mqtt_client_handle_t client = NULL;
static naila_mqtt_message_handler_t message_handler = NULL;
static bool connected = false;

static void mqtt_event_handler(void* handler_args, esp_event_base_t base, int32_t event_id, void* event_data) {
    esp_mqtt_event_handle_t event = event_data;

    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "✓ Successfully connected to MQTT broker at %s:%d", CONFIG_MQTT_BROKER_IP, CONFIG_MQTT_BROKER_PORT);
            ESP_LOGI(TAG, "✓ Client ID: %s", CONFIG_ROBOT_ID);
            connected = true;
            break;

        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "✗ Disconnected from MQTT broker");
            connected = false;
            break;

        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "✓ Successfully subscribed to topic (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_UNSUBSCRIBED:
            ESP_LOGI(TAG, "✓ Successfully unsubscribed from topic (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_PUBLISHED:
            ESP_LOGI(TAG, "✓ Message published successfully (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "✓ Received message on topic: %.*s", event->topic_len, event->topic);
            if (message_handler) {
                char topic[128] = {0};
                int topic_len = event->topic_len < sizeof(topic) - 1 ? event->topic_len : sizeof(topic) - 1;
                strncpy(topic, event->topic, topic_len);
                message_handler(topic, event->data, event->data_len);
            }
            break;

        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "✗ MQTT error occurred:");
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGE(TAG, "  - TCP transport error: 0x%x", event->error_handle->esp_transport_sock_errno);
                ESP_LOGE(TAG, "  - ESP-TLS error: 0x%x", event->error_handle->esp_tls_last_esp_err);
                ESP_LOGE(TAG, "  - TLS stack error: 0x%x", event->error_handle->esp_tls_stack_err);
            } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                ESP_LOGE(TAG, "  - Connection refused by broker");
            }
            break;

        case MQTT_EVENT_BEFORE_CONNECT:
            ESP_LOGI(TAG, "⏳ Attempting to connect to MQTT broker...");
            break;

        default:
            ESP_LOGD(TAG, "Unhandled MQTT event: %d", event->event_id);
            break;
    }
}

esp_err_t naila_mqtt_init(void) {
    if (client != NULL) {
        ESP_LOGI(TAG, "MQTT client already initialized, skipping...");
        return ESP_OK;
    }

    ESP_LOGI(TAG, "🚀 Initializing MQTT client...");

    char mqtt_uri[64];
    snprintf(mqtt_uri, sizeof(mqtt_uri), "mqtt://%s:%d", CONFIG_MQTT_BROKER_IP, CONFIG_MQTT_BROKER_PORT);

    ESP_LOGI(TAG, "📡 Broker URI: %s", mqtt_uri);
    ESP_LOGI(TAG, "🆔 Client ID: %s", CONFIG_ROBOT_ID);
    ESP_LOGI(TAG, "⏱️  Timeouts: reconnect=%dms, network=%dms", 10000, 10000);

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = "mqtt://10.0.0.117:1883",  // Use mqtt:// not mqtts://
                // OR alternatively use hostname and port:
                // .hostname = "10.0.0.117",
                // .port = 1883,
                // .transport = MQTT_TRANSPORT_OVER_TCP,
            },
        },
        .credentials = {
            .client_id = CONFIG_ROBOT_ID,
        },
        .session = {
            .keepalive = 60,
            .protocol_ver = MQTT_PROTOCOL_V_3_1_1,
        },
        .network = {
            .reconnect_timeout_ms = 10000,
            .timeout_ms = 10000,
        },
    };

    ESP_LOGI(TAG, "⚙️  Creating MQTT client instance...");
    client = esp_mqtt_client_init(&mqtt_cfg);
    if (client == NULL) {
        ESP_LOGE(TAG, "✗ Failed to initialize MQTT client");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "✓ MQTT client instance created");

    ESP_LOGI(TAG, "📋 Registering event handler...");
    esp_err_t ret = esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "✗ Failed to register MQTT event handler: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "✓ Event handler registered");

    ESP_LOGI(TAG, "🔌 Starting MQTT client...");
    ret = esp_mqtt_client_start(client);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "✗ Failed to start MQTT client: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "✅ MQTT client initialized and started successfully");
    ESP_LOGI(TAG, "🔄 Client will attempt to connect to broker...");
    return ESP_OK;
}

esp_err_t naila_mqtt_publish(const char* topic, const char* data, int len, int qos) {
    if (!connected || !client) {
        ESP_LOGW(TAG, "✗ Cannot publish: MQTT client not connected");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "📤 Publishing to topic '%s' (QoS %d, len %d)", topic, qos, len);
    int msg_id = esp_mqtt_client_publish(client, topic, data, len, qos, 0);
    if (msg_id < 0) {
        ESP_LOGE(TAG, "✗ Failed to publish message to topic '%s'", topic);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "✓ Message queued for publish (msg_id: %d)", msg_id);
    return ESP_OK;
}

esp_err_t naila_mqtt_subscribe(const char* topic, int qos) {
    if (!connected || !client) {
        ESP_LOGW(TAG, "✗ Cannot subscribe: MQTT client not connected");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "📥 Subscribing to topic '%s' (QoS %d)", topic, qos);
    int msg_id = esp_mqtt_client_subscribe(client, topic, qos);
    if (msg_id < 0) {
        ESP_LOGE(TAG, "✗ Failed to subscribe to topic '%s'", topic);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "✓ Subscription request sent (msg_id: %d)", msg_id);
    return ESP_OK;
}

void naila_mqtt_register_handler(naila_mqtt_message_handler_t handler) {
    message_handler = handler;
}

bool naila_mqtt_is_connected(void) {
    return connected;
}