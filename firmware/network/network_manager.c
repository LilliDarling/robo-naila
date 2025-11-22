#include "network_manager.h"
#include "wifi.h"
#include "mqtt_client.h"
#include "config.h"
#include "naila_log.h"
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <string.h>

static const char *TAG = "NETWORK_MANAGER";

typedef struct {
    bool wifi_connected;
    bool mqtt_connected;
    bool initialized;
    network_event_callback_t callback;
} network_manager_t;

static network_manager_t g_network = {0};
static SemaphoreHandle_t g_state_mutex = NULL;

// Forward declarations
static void on_wifi_ready(void);
static void on_wifi_lost(void);
static void on_wifi_error(naila_err_t error);

naila_err_t network_manager_init(network_config_t* config) {
    // Create mutex on first init
    if (!g_state_mutex) {
        g_state_mutex = xSemaphoreCreateMutex();
        if (!g_state_mutex) {
            NAILA_LOGE(TAG, "Failed to create state mutex");
            return NAILA_ERR_NO_MEM;
        }
    }

    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        if (g_network.initialized) {
            xSemaphoreGive(g_state_mutex);
            NAILA_LOGW(TAG, "Network manager already initialized");
            return NAILA_ERR_ALREADY_INITIALIZED;
        }
        g_network.initialized = true;
        g_network.callback = config->callback;
        xSemaphoreGive(g_state_mutex);
    }

    naila_err_t wifi_result = wifi_init();
    if (wifi_result != NAILA_OK) {
        if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
            g_network.initialized = false;
            xSemaphoreGive(g_state_mutex);
        }
        NAILA_LOG_ERROR(TAG, wifi_result, "Error in wifi init: wifi_init()");
        return wifi_result;
    }

    NAILA_LOGI(TAG, "Network manager initialized");
    return NAILA_OK;
}

naila_err_t network_manager_start(void) {
    bool is_initialized = false;
    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        is_initialized = g_network.initialized;
        xSemaphoreGive(g_state_mutex);
    }

    if (!is_initialized) {
        NAILA_LOGE(TAG, "Network manager not initialized");
        return NAILA_ERR_NOT_INITIALIZED;
    }

    if (wifi_is_task_running()) {
        NAILA_LOGW(TAG, "Network manager already started");
        return NAILA_OK;
    }

    wifi_event_callbacks_t wifi_cb = {
        .on_connected = on_wifi_ready,
        .on_disconnected = on_wifi_lost,
        .on_error = on_wifi_error,
        .on_state_change = NULL
    };

    naila_err_t err = wifi_start_task(&wifi_cb);
    if (err != NAILA_OK) {
        NAILA_LOGE(TAG, "Failed to start WiFi task: 0x%x", err);
        return err;
    }

    NAILA_LOGI(TAG, "Network manager started");
    return NAILA_OK;
}

naila_err_t network_manager_stop(void) {
    if (!g_state_mutex) {
        return NAILA_OK;
    }

    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        if (!g_network.initialized) {
            xSemaphoreGive(g_state_mutex);
            return NAILA_OK;
        }

        g_network.initialized = false;
        g_network.callback = NULL;
        g_network.wifi_connected = false;
        g_network.mqtt_connected = false;
        xSemaphoreGive(g_state_mutex);
    }

    mqtt_client_stop();
    wifi_stop_task();

    NAILA_LOGI(TAG, "Network manager stopped");
    return NAILA_OK;
}

bool network_manager_is_ready(void) {
    bool ready = false;
    if (g_state_mutex && xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        ready = g_network.wifi_connected && g_network.mqtt_connected;
        xSemaphoreGive(g_state_mutex);
    }
    return ready;
}

bool network_manager_is_wifi_connected(void) {
    bool connected = false;
    if (g_state_mutex && xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        connected = g_network.wifi_connected;
        xSemaphoreGive(g_state_mutex);
    }
    return connected;
}

bool network_manager_is_mqtt_connected(void) {
    bool connected = false;
    if (g_state_mutex && xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        connected = g_network.mqtt_connected;
        xSemaphoreGive(g_state_mutex);
    }
    return connected;
}

// WiFi callback implementations
static void on_wifi_ready(void) {
    NAILA_LOGI(TAG, "WiFi connected");

    network_event_callback_t callback = NULL;
    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        g_network.wifi_connected = true;
        callback = g_network.callback;
        xSemaphoreGive(g_state_mutex);
    }

    if (callback) {
        callback(NETWORK_EVENT_WIFI_CONNECTED);
    }

    const naila_config_t *config = config_manager_get();
    if (!config) {
        NAILA_LOGE(TAG, "Failed to get config from config manager");
        if (callback) callback(NETWORK_EVENT_ERROR);
        return;
    }

    naila_err_t result = mqtt_client_init(&config->mqtt);
    if (result == NAILA_OK) {
        NAILA_LOGI(TAG, "MQTT connected");

        if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
            g_network.mqtt_connected = true;
            callback = g_network.callback;
            xSemaphoreGive(g_state_mutex);
        }

        if (callback) {
            callback(NETWORK_EVENT_MQTT_CONNECTED);
            callback(NETWORK_EVENT_CONTROL_PLANE_READY);
        }
    } else {
        NAILA_LOGE(TAG, "MQTT initialization failed: 0x%x", result);
        if (callback) callback(NETWORK_EVENT_ERROR);
    }
}

static void on_wifi_lost(void) {
    NAILA_LOGI(TAG, "WiFi disconnected");

    network_event_callback_t callback = NULL;
    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        g_network.wifi_connected = false;
        g_network.mqtt_connected = false;
        callback = g_network.callback;
        xSemaphoreGive(g_state_mutex);
    }

    if (callback) {
        callback(NETWORK_EVENT_WIFI_DISCONNECTED);
    }
}

static void on_wifi_error(naila_err_t error) {
    NAILA_LOGE(TAG, "WiFi error: 0x%x", error);

    network_event_callback_t callback = NULL;
    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY)) {
        callback = g_network.callback;
        xSemaphoreGive(g_state_mutex);
    }

    if (callback) {
        callback(NETWORK_EVENT_ERROR);
    }
}