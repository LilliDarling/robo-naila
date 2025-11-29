#include "app.h"
#include "config.h"
#include "esp_system.h"
#include "esp_timer.h"
#include <esp_task_wdt.h>
#include <freertos/semphr.h>

#include "wifi.h"
#include "naila_mqtt.h"
#include "naila_log.h"
#include "mutex_helpers.h"

static const char *TAG = "APP_MANAGER";

// Configuration constants
#define WATCHDOG_TIMEOUT_SEC 30
#define WATCHDOG_PANIC_ON_TIMEOUT true

// Unified application manager state
typedef struct {
    app_state_t state;
    const app_callbacks_t *callbacks;
} app_manager_t;

static app_manager_t g_app = {0};
static SemaphoreHandle_t g_app_mutex = NULL;

// Forward declarations
static void on_wifi_connected(void);
static void on_wifi_error(naila_err_t error);

naila_err_t app_manager_init(const app_callbacks_t *callbacks) {
  if (!g_app_mutex) {
    g_app_mutex = xSemaphoreCreateMutex();
    if (!g_app_mutex) {
      NAILA_LOGE(TAG, "Failed to create app mutex");
      return NAILA_ERR_NO_MEM;
    }
  }

  MUTEX_LOCK(g_app_mutex, TAG) {
    g_app.callbacks = callbacks;
    g_app.state = APP_STATE_INITIALIZING;
  } MUTEX_UNLOCK(g_app_mutex, TAG);

  naila_err_t result = config_manager_init();
  if (result != NAILA_OK) {
    NAILA_LOG_ERROR(TAG, result, "Error in config manager init: config_manager_init()");
    return result;
  }

  result = wifi_init();
  if (result != NAILA_OK) {
    NAILA_LOG_ERROR(TAG, result, "Error in wifi init: wifi_init()");
    return result;
  }

  NAILA_LOGI(TAG, "Application manager initialized");
  return NAILA_OK;
}

// WiFi callback implementations
static void on_wifi_connected(void) {
  NAILA_LOGI(TAG, "WiFi connected");

  const app_callbacks_t *callbacks = NULL;
  MUTEX_LOCK_VOID(g_app_mutex, TAG) {
    g_app.state = APP_STATE_SERVICES_STARTING;
    callbacks = g_app.callbacks;
  } MUTEX_UNLOCK_VOID(g_app_mutex, TAG);

  if (callbacks && callbacks->on_state_change) {
    callbacks->on_state_change(APP_STATE_SERVICES_STARTING);
  }

  const naila_config_t *config = config_manager_get();
  if (!config) {
    NAILA_LOGE(TAG, "Failed to get config from config manager");
    on_wifi_error(NAILA_ERR_INVALID_ARG);
    return;
  }

  naila_err_t result = mqtt_client_init(&config->mqtt);
  if (result == NAILA_OK) {
    NAILA_LOGI(TAG, "MQTT connected - control plane ready");

    MUTEX_LOCK_VOID(g_app_mutex, TAG) {
      g_app.state = APP_STATE_RUNNING;
      callbacks = g_app.callbacks;
    } MUTEX_UNLOCK_VOID(g_app_mutex, TAG);

    if (callbacks && callbacks->on_state_change) {
      callbacks->on_state_change(APP_STATE_RUNNING);
    }
  } else {
    NAILA_LOGE(TAG, "MQTT initialization failed: 0x%x", result);
    on_wifi_error(result);
  }
}

static void on_wifi_error(naila_err_t error) {
  NAILA_LOGE(TAG, "WiFi error: 0x%x", error);

  const app_callbacks_t *callbacks = NULL;
  MUTEX_LOCK_VOID(g_app_mutex, TAG) {
    g_app.state = APP_STATE_ERROR;
    callbacks = g_app.callbacks;
  } MUTEX_UNLOCK_VOID(g_app_mutex, TAG);

  if (callbacks && callbacks->on_state_change) {
    callbacks->on_state_change(APP_STATE_ERROR);
  }
  if (callbacks && callbacks->on_error) {
    callbacks->on_error(error);
  }
}

naila_err_t app_manager_start(void) {
  esp_task_wdt_config_t wdt_config = {
    .timeout_ms = WATCHDOG_TIMEOUT_SEC * 1000,
    .idle_core_mask = 0,
    .trigger_panic = WATCHDOG_PANIC_ON_TIMEOUT,
  };
  esp_err_t wdt_result = esp_task_wdt_init(&wdt_config);
  if (wdt_result != ESP_OK) {
    NAILA_LOGE(TAG, "Failed to initialize watchdog: 0x%x", wdt_result);
    return (naila_err_t)wdt_result;
  }

  wifi_event_callbacks_t wifi_cb = {
    .on_connected = on_wifi_connected,
    .on_error = on_wifi_error
  };

  naila_err_t err = wifi_start_task(&wifi_cb);
  if (err != NAILA_OK) {
    NAILA_LOGE(TAG, "Failed to start WiFi task: 0x%x", err);
    return err;
  }

  NAILA_LOGI(TAG, "Application started");
  return NAILA_OK;
}

naila_err_t app_manager_stop(void) {
  MUTEX_LOCK(g_app_mutex, TAG) {
    g_app.callbacks = NULL;
    g_app.state = APP_STATE_SHUTDOWN;
  } MUTEX_UNLOCK(g_app_mutex, TAG);

  mqtt_client_stop();
  wifi_stop_task();

  NAILA_LOGI(TAG, "Application stopped");
  return NAILA_OK;
}

app_state_t app_manager_get_state(void) {
  app_state_t state = APP_STATE_ERROR;
  MUTEX_LOCK(g_app_mutex, TAG) {
    state = g_app.state;
  } MUTEX_UNLOCK(g_app_mutex, TAG);
  return state;
}

bool app_manager_is_running(void) {
  bool running = false;
  MUTEX_LOCK_BOOL(g_app_mutex, TAG) {
    running = (g_app.state == APP_STATE_RUNNING);
  } MUTEX_UNLOCK_BOOL(g_app_mutex, TAG);
  return running;
}