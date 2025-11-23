#include "network_manager.h"
#include "wifi.h"
#include "mqtt_client.h"
#include "config.h"
#include "naila_log.h"
#include "mutex_helpers.h"
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

    bool already_initialized = false;
    MUTEX_LOCK(g_state_mutex, TAG) {
        already_initialized = g_network.initialized;
        if (!already_initialized) {
            g_network.initialized = true;
            g_network.callback = config->callback;
        }
    } MUTEX_UNLOCK();

    if (already_initialized) {
        NAILA_LOGW(TAG, "Network manager already initialized");
        return NAILA_ERR_ALREADY_INITIALIZED;
    }

    naila_err_t wifi_result = wifi_init();
    if (wifi_result != NAILA_OK) {
        MUTEX_LOCK(g_state_mutex, TAG) {
            g_network.initialized = false;
        } MUTEX_UNLOCK();
        NAILA_LOG_ERROR(TAG, wifi_result, "Error in wifi init: wifi_init()");
        return wifi_result;
    }

    NAILA_LOGI(TAG, "Network manager initialized");
    return NAILA_OK;
}

naila_err_t network_manager_start(void) {
    bool is_initialized = false;
    MUTEX_LOCK(g_state_mutex, TAG) {
        is_initialized = g_network.initialized;
    } MUTEX_UNLOCK();

    if (!is_initialized) {
        NAILA_LOGE(TAG, "Network manager not initialized");
        return NAILA_ERR_NOT_INITIALIZED;
    }

    wifi_event_callbacks_t wifi_cb = {
        .on_connected = on_wifi_ready,
        .on_error = on_wifi_error
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

    bool is_initialized = false;
    MUTEX_LOCK(g_state_mutex, TAG) {
        is_initialized = g_network.initialized;
        if (is_initialized) {
            g_network.initialized = false;
            g_network.callback = NULL;
            g_network.wifi_connected = false;
            g_network.mqtt_connected = false;
        }
    } MUTEX_UNLOCK();

    if (!is_initialized) {
        return NAILA_OK;
    }

    mqtt_client_stop();
    wifi_stop_task();

    NAILA_LOGI(TAG, "Network manager stopped");
    return NAILA_OK;
}

bool network_manager_is_ready(void) {
    bool ready = false;
    if (g_state_mutex) {
        MUTEX_LOCK_BOOL(g_state_mutex, TAG) {
            ready = g_network.wifi_connected && g_network.mqtt_connected;
        } MUTEX_UNLOCK_BOOL();
    }
    return ready;
}

bool network_manager_is_wifi_connected(void) {
    bool connected = false;
    if (g_state_mutex) {
        MUTEX_LOCK_BOOL(g_state_mutex, TAG) {
            connected = g_network.wifi_connected;
        } MUTEX_UNLOCK_BOOL();
    }
    return connected;
}

bool network_manager_is_mqtt_connected(void) {
    bool connected = false;
    if (g_state_mutex) {
        MUTEX_LOCK_BOOL(g_state_mutex, TAG) {
            connected = g_network.mqtt_connected;
        } MUTEX_UNLOCK_BOOL();
    }
    return connected;
}

// WiFi callback implementations
static void on_wifi_ready(void) {
    NAILA_LOGI(TAG, "WiFi connected");

    network_event_callback_t callback = NULL;
    MUTEX_LOCK_VOID(g_state_mutex, TAG) {
        g_network.wifi_connected = true;
        callback = g_network.callback;
    } MUTEX_UNLOCK_VOID();

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

        MUTEX_LOCK_VOID(g_state_mutex, TAG) {
            g_network.mqtt_connected = true;
            callback = g_network.callback;
        } MUTEX_UNLOCK_VOID();

        if (callback) {
            callback(NETWORK_EVENT_MQTT_CONNECTED);
            callback(NETWORK_EVENT_CONTROL_PLANE_READY);
        }
    } else {
        NAILA_LOGE(TAG, "MQTT initialization failed: 0x%x", result);
        if (callback) callback(NETWORK_EVENT_ERROR);
    }
}

static void on_wifi_error(naila_err_t error) {
    NAILA_LOGE(TAG, "WiFi error: 0x%x", error);

    network_event_callback_t callback = NULL;
    MUTEX_LOCK_VOID(g_state_mutex, TAG) {
        callback = g_network.callback;
    } MUTEX_UNLOCK_VOID();

    if (callback) {
        callback(NETWORK_EVENT_ERROR);
    }
}