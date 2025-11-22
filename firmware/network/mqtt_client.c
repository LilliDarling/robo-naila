#include "mqtt_client.h"
#include "naila_log.h"
#include <stdio.h>
#include <string.h>
#include <freertos/semphr.h>

static const char* TAG = "MQTT_CLIENT";

// MQTT client configuration constants
#define MQTT_RECONNECT_TIMEOUT_MS 10000
#define MQTT_NETWORK_TIMEOUT_MS 10000

static esp_mqtt_client_handle_t client = NULL;
static mqtt_message_handler_t message_handler = NULL;
static bool connected = false;
static bool initialized = false;
static SemaphoreHandle_t mqtt_mutex = NULL;
static char mqtt_uri_buffer[64];

static void mqtt_event_handler(void* handler_args __attribute__((unused)),
                               esp_event_base_t base __attribute__((unused)),
                               int32_t event_id,
                               void* event_data) {
    if (!event_data) {
        NAILA_LOGW(TAG, "Received NULL event_data in MQTT event handler");
        return;
    }

    esp_mqtt_event_handle_t event = event_data;

    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            NAILA_LOGI(TAG, "Successfully connected to MQTT broker");
            if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
                connected = true;
                xSemaphoreGive(mqtt_mutex);
            }
            break;

        case MQTT_EVENT_DISCONNECTED:
            NAILA_LOGW(TAG, "Disconnected from MQTT broker");
            if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
                connected = false;
                xSemaphoreGive(mqtt_mutex);
            }
            break;

        case MQTT_EVENT_SUBSCRIBED:
            NAILA_LOGI(TAG, "Successfully subscribed to topic (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_UNSUBSCRIBED:
            NAILA_LOGI(TAG, "Successfully unsubscribed from topic (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_PUBLISHED:
            NAILA_LOGI(TAG, "Message published successfully (msg_id: %d)", event->msg_id);
            break;

        case MQTT_EVENT_DATA:
            NAILA_LOGI(TAG, "Received message on topic: %.*s", event->topic_len, event->topic);
            if (event->topic_len > 0) {
                mqtt_message_handler_t handler = NULL;
                if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
                    handler = message_handler;
                    xSemaphoreGive(mqtt_mutex);
                }

                if (handler) {
                    char* topic = malloc(event->topic_len + 1);
                    if (topic) {
                        memcpy(topic, event->topic, event->topic_len);
                        topic[event->topic_len] = '\0';
                        handler(topic, event->data, event->data_len);
                        free(topic);
                    } else {
                        NAILA_LOGE(TAG, "Failed to allocate memory for topic (%d bytes)", event->topic_len + 1);
                    }
                }
            }
            break;

        case MQTT_EVENT_ERROR:
            NAILA_LOGE(TAG, "MQTT error occurred");
            if (event->error_handle) {
                if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                    NAILA_LOGE(TAG, "  - TCP transport error: 0x%x", event->error_handle->esp_transport_sock_errno);
                    NAILA_LOGE(TAG, "  - ESP-TLS error: 0x%x", event->error_handle->esp_tls_last_esp_err);
                    NAILA_LOGE(TAG, "  - TLS stack error: 0x%x", event->error_handle->esp_tls_stack_err);
                } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                    NAILA_LOGE(TAG, "  - Connection refused by broker");
                }
            } else {
                NAILA_LOGE(TAG, "  - No error details available (error_handle is NULL)");
            }
            break;

        case MQTT_EVENT_BEFORE_CONNECT:
            NAILA_LOGI(TAG, "Attempting to connect to MQTT broker...");
            break;

        default:
            NAILA_LOGD(TAG, "Unhandled MQTT event: %d", event->event_id);
            break;
    }
}

naila_err_t mqtt_client_init(const mqtt_config_t* config) {
    // Create mutex on first init
    if (!mqtt_mutex) {
        mqtt_mutex = xSemaphoreCreateMutex();
        if (!mqtt_mutex) {
            NAILA_LOGE(TAG, "Failed to create MQTT mutex");
            return NAILA_ERR_NO_MEM;
        }
    }

    if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
        if (initialized) {
            xSemaphoreGive(mqtt_mutex);
            NAILA_LOGW(TAG, "MQTT client already initialized");
            return NAILA_OK;
        }
        initialized = true;
        xSemaphoreGive(mqtt_mutex);
    }

    snprintf(mqtt_uri_buffer, sizeof(mqtt_uri_buffer), "mqtt://%s:%d", config->broker_ip, config->broker_port);

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = mqtt_uri_buffer,
        .credentials.client_id = config->robot_id,
        .session.keepalive = config->keepalive_sec,
        .session.protocol_ver = MQTT_PROTOCOL_V_3_1_1,
        .network.reconnect_timeout_ms = MQTT_RECONNECT_TIMEOUT_MS,
        .network.timeout_ms = MQTT_NETWORK_TIMEOUT_MS,
    };

    client = esp_mqtt_client_init(&mqtt_cfg);
    if (!client) {
        NAILA_LOGE(TAG, "Failed to initialize MQTT client");
        if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
            initialized = false;
            xSemaphoreGive(mqtt_mutex);
        }
        return NAILA_FAIL;
    }

    esp_err_t ret = esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    if (ret != ESP_OK) {
        NAILA_LOGE(TAG, "Failed to register MQTT event handler: 0x%x", ret);
        esp_mqtt_client_destroy(client);
        client = NULL;
        if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
            initialized = false;
            xSemaphoreGive(mqtt_mutex);
        }
        return (naila_err_t)ret;
    }

    ret = esp_mqtt_client_start(client);
    if (ret != ESP_OK) {
        NAILA_LOGE(TAG, "Failed to start MQTT client: 0x%x", ret);
        esp_mqtt_client_destroy(client);
        client = NULL;
        if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
            initialized = false;
            xSemaphoreGive(mqtt_mutex);
        }
        return (naila_err_t)ret;
    }

    NAILA_LOGI(TAG, "MQTT client initialized");
    return NAILA_OK;
}

naila_err_t mqtt_client_publish(const char* topic, const char* data, int len, int qos) {
    if (!mqtt_client_is_connected()) {
        NAILA_LOGW(TAG, "MQTT client not connected");
        return NAILA_ERR_INVALID_ARG;
    }

    if (!client) {
        NAILA_LOGW(TAG, "MQTT client not initialized");
        return NAILA_ERR_INVALID_ARG;
    }

    int msg_id = esp_mqtt_client_publish(client, topic, data, len, qos, 0);
    if (msg_id < 0) {
        NAILA_LOGE(TAG, "Failed to publish to '%s'", topic);
        return NAILA_FAIL;
    }

    return NAILA_OK;
}

naila_err_t mqtt_client_subscribe(const char* topic, int qos) {
    if (!mqtt_client_is_connected()) {
        NAILA_LOGW(TAG, "MQTT client not connected");
        return NAILA_ERR_INVALID_ARG;
    }

    if (!client) {
        NAILA_LOGW(TAG, "MQTT client not initialized");
        return NAILA_ERR_INVALID_ARG;
    }

    int msg_id = esp_mqtt_client_subscribe(client, topic, qos);
    if (msg_id < 0) {
        NAILA_LOGE(TAG, "Failed to subscribe to '%s'", topic);
        return NAILA_FAIL;
    }

    return NAILA_OK;
}

void mqtt_client_register_handler(mqtt_message_handler_t handler) {
    if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
        message_handler = handler;
        xSemaphoreGive(mqtt_mutex);
    }
}

bool mqtt_client_is_connected(void) {
    bool status = false;
    if (mqtt_mutex && xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
        status = connected;
        xSemaphoreGive(mqtt_mutex);
    }
    return status;
}

naila_err_t mqtt_client_stop(void) {
    if (!mqtt_mutex) {
        return NAILA_OK;
    }

    if (xSemaphoreTake(mqtt_mutex, portMAX_DELAY)) {
        if (!initialized) {
            xSemaphoreGive(mqtt_mutex);
            return NAILA_OK;
        }

        initialized = false;
        connected = false;
        message_handler = NULL;

        if (client) {
            esp_err_t ret = esp_mqtt_client_stop(client);
            if (ret != ESP_OK) {
                NAILA_LOGW(TAG, "MQTT client stop returned: 0x%x", ret);
            }

            ret = esp_mqtt_client_destroy(client);
            if (ret != ESP_OK) {
                NAILA_LOGW(TAG, "MQTT client destroy returned: 0x%x", ret);
            }
            client = NULL;
        }

        xSemaphoreGive(mqtt_mutex);
    }

    NAILA_LOGI(TAG, "MQTT client stopped");
    return NAILA_OK;
}
