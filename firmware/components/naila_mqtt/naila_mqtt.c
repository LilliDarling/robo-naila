#include "naila_mqtt.h"
#include "mutex_utils.h"
#include "esp_log.h"
#include "mqtt_client.h"
#include <stdio.h>
#include <string.h>
#include <freertos/semphr.h>

static const char* TAG = "NAILA_MQTT";
static esp_mqtt_client_handle_t client = NULL;
static naila_mqtt_message_handler_t message_handler = NULL;
static bool connected = false;

static SemaphoreHandle_t mqtt_mutex = NULL;

static void mqtt_event_handler(void* handler_args, esp_event_base_t base, int32_t event_id, void* event_data) {
    esp_mqtt_event_handle_t event = event_data;

    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "Successfully connected to MQTT broker");
            mutex_execute(mqtt_mutex, ^(void* ctx) {
                connected = true;
                return NAILA_OK;
            }, NULL);
            break;

        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "Disconnected from MQTT broker");
            mutex_execute(mqtt_mutex, ^(void* ctx) {
                connected = false;
                return NAILA_OK;
            }, NULL);
            break;

        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "Successfully subscribed to topic (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_UNSUBSCRIBED:
            ESP_LOGI(TAG, "Successfully unsubscribed from topic (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_PUBLISHED:
            ESP_LOGI(TAG, "Message published successfully (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "Received message on topic: %.*s", event->topic_len, event->topic);
            if (message_handler && event->topic_len > 0) {
                char* topic = malloc(event->topic_len + 1);
                if (topic) {
                    memcpy(topic, event->topic, event->topic_len);
                    topic[event->topic_len] = '\0';
                    message_handler(topic, event->data, event->data_len);
                    free(topic);
                }
            }
            break;

        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT error occurred:");
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGE(TAG, "  - TCP transport error: 0x%x", event->error_handle->esp_transport_sock_errno);
                ESP_LOGE(TAG, "  - ESP-TLS error: 0x%x", event->error_handle->esp_tls_last_esp_err);
                ESP_LOGE(TAG, "  - TLS stack error: 0x%x", event->error_handle->esp_tls_stack_err);
            } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                ESP_LOGE(TAG, "  - Connection refused by broker");
            }
            break;

        case MQTT_EVENT_BEFORE_CONNECT:
            ESP_LOGI(TAG, "Attempting to connect to MQTT broker...");
            break;

        default:
            ESP_LOGD(TAG, "Unhandled MQTT event: %d", event->event_id);
            break;
    }
}

esp_err_t naila_mqtt_init(const mqtt_config_t* config) {
    if (client) return ESP_OK;
    if (!config) return ESP_ERR_INVALID_ARG;

    mqtt_mutex = xSemaphoreCreateMutex();
    if (!mqtt_mutex) {
        ESP_LOGE(TAG, "Failed to create MQTT mutex");
        return ESP_FAIL;
    }

    char mqtt_uri[64];
    snprintf(mqtt_uri, sizeof(mqtt_uri), "mqtt://%s:%d", config->broker_ip, config->broker_port);

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = mqtt_uri,
        .credentials.client_id = config->robot_id,
        .session.keepalive = config->keepalive_sec,
        .session.protocol_ver = MQTT_PROTOCOL_V_3_1_1,
        .network.reconnect_timeout_ms = 10000,
        .network.timeout_ms = 10000,
    };

    client = esp_mqtt_client_init(&mqtt_cfg);
    if (!client) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return ESP_FAIL;
    }

    esp_err_t ret = esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register MQTT event handler: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_mqtt_client_start(client);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start MQTT client: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "MQTT client initialized");
    return ESP_OK;
}

esp_err_t naila_mqtt_publish(const char* topic, const char* data, int len, int qos) {
    if (!naila_mqtt_is_connected() || !client) return ESP_ERR_INVALID_STATE;

    int msg_id = esp_mqtt_client_publish(client, topic, data, len, qos, 0);
    if (msg_id < 0) {
        ESP_LOGE(TAG, "Failed to publish to '%s'", topic);
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t naila_mqtt_subscribe(const char* topic, int qos) {
    if (!naila_mqtt_is_connected() || !client) return ESP_ERR_INVALID_STATE;

    int msg_id = esp_mqtt_client_subscribe(client, topic, qos);
    if (msg_id < 0) {
        ESP_LOGE(TAG, "Failed to subscribe to '%s'", topic);
        return ESP_FAIL;
    }

    return ESP_OK;
}

void naila_mqtt_register_handler(naila_mqtt_message_handler_t handler) {
    message_handler = handler;
}

bool naila_mqtt_is_connected(void) {
    if (!mqtt_mutex) return connected;

    bool status;
    mutex_execute(mqtt_mutex, ^(void* ctx) {
        status = connected;
        return NAILA_OK;
    }, NULL);

    return status;
}