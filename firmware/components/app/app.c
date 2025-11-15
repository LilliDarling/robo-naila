#include "app.h"
#include "config.h"
#include "error_handling.h"
#include "esp_system.h"
#include "esp_timer.h"
#include <esp_task_wdt.h>
#include <freertos/semphr.h>

#include "naila_mqtt.h"
#include "naila_log.h"
#include "wifi.h"
#include <string.h>

static TaskHandle_t wifi_task_handle = NULL;
static TaskHandle_t stats_task_handle = NULL;

static SemaphoreHandle_t state_mutex = NULL;
static SemaphoreHandle_t stats_mutex = NULL;

static const char *TAG = "app_manager";
static const app_callbacks_t *g_callbacks = NULL;

static component_state_t g_component_state = COMPONENT_STATE_UNINITIALIZED;
static app_state_t g_app_state = APP_STATE_INITIALIZING;
static app_stats_t g_stats = {0};

static const TickType_t MUTEX_TIMEOUT = pdMS_TO_TICKS(100);


static const char *app_state_to_string(app_state_t state) {
  switch (state) {
  case APP_STATE_INITIALIZING:
    return "INITIALIZING";
  case APP_STATE_WIFI_CONNECTING:
    return "WIFI_CONNECTING";
  case APP_STATE_SERVICES_STARTING:
    return "SERVICES_STARTING";
  case APP_STATE_RUNNING:
    return "RUNNING";
  case APP_STATE_ERROR:
    return "ERROR";
  case APP_STATE_SHUTDOWN:
    return "SHUTDOWN";
  default:
    return "UNKNOWN";
  }
}

naila_err_t app_manager_set_state(app_state_t new_state) {
  // Take mutex with timeout
  if (xSemaphoreTake(state_mutex, MUTEX_TIMEOUT) != pdTRUE) {
    NAILA_LOGE(TAG, "Failed to acquire state mutex");
    return NAILA_ERR_TIMEOUT;
  }
  if (new_state == g_app_state) {
    xSemaphoreGive(state_mutex);
    return NAILA_OK;
  }

  app_state_t old_state = g_app_state;
  g_app_state = new_state;

  xSemaphoreGive(state_mutex);

  NAILA_LOGI(TAG, "State transition: %s -> %s", app_state_to_string(old_state),
      app_state_to_string(new_state));

  if (g_callbacks && g_callbacks->on_state_change) {
    g_callbacks->on_state_change(old_state, new_state);
  }

  return NAILA_OK;
}

naila_err_t app_manager_init(const app_callbacks_t *callbacks) {
  NAILA_LOG_FUNC_ENTER(TAG);

  if (g_component_state == COMPONENT_STATE_INITIALIZED) {
    return NAILA_ERR_ALREADY_INITIALIZED;
  }

  state_mutex = xSemaphoreCreateMutex();
  if (!state_mutex) {
    NAILA_LOGE(TAG, "Failed to create state mutex");
    return NAILA_ERR_NO_MEMORY;
  }

  stats_mutex = xSemaphoreCreateMutex();
  if (!stats_mutex) {
    NAILA_LOGE(TAG, "Failed to create stats mutex");
    return NAILA_ERR_NO_MEMORY;
  }

  g_component_state = COMPONENT_STATE_INITIALIZING;

  g_callbacks = callbacks;

  // Initialize configuration manager
  NAILA_PROPAGATE_ERROR(config_manager_init(), TAG, "config manager init");

  // Initialize WiFi manager
  NAILA_PROPAGATE_ERROR(wifi_init(), TAG, "wifi init");

  // Initialize statistics
  memset(&g_stats, 0, sizeof(app_stats_t));

  g_component_state = COMPONENT_STATE_INITIALIZED;
  NAILA_LOGI(TAG, "Application manager initialized");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

// WiFi event callbacks for the app_manager
static void on_wifi_connected(void) {

  NAILA_LOGI(TAG, "🎉 WiFi connected callback triggered");
  app_manager_set_state(APP_STATE_SERVICES_STARTING);

  // Initialize MQTT after WiFi connection
  naila_err_t result = naila_mqtt_init();
  if (result == NAILA_OK) {
    NAILA_LOGI(TAG, "MQTT initialized after WiFi connection");
    app_manager_set_state(APP_STATE_RUNNING);
  } else {
    NAILA_LOGE(TAG, "MQTT initialization failed");
    app_manager_set_state(APP_STATE_ERROR);
  }

  if (g_callbacks && g_callbacks->on_wifi_connected) {
    g_callbacks->on_wifi_connected();
  }
}

static void on_wifi_disconnected(void) {
  if (xSemaphoreTake(stats_mutex, MUTEX_TIMEOUT) == pdTRUE) {
    g_stats.wifi_reconnect_count++;
    xSemaphoreGive(stats_mutex);
  }

  app_manager_set_state(APP_STATE_WIFI_CONNECTING);
  
  if (g_callbacks && g_callbacks->on_wifi_disconnected) {
    g_callbacks->on_wifi_disconnected();
  }
}

static void on_wifi_error(naila_err_t error) {
  if (xSemaphoreTake(stats_mutex, MUTEX_TIMEOUT) == pdTRUE) {
    g_stats.error_count++;
    xSemaphoreGive(stats_mutex);
  }

  app_manager_set_state(APP_STATE_ERROR);

  if (g_callbacks && g_callbacks->on_error) {
    g_callbacks->on_error(error);
  }
}

static void on_wifi_state_change(int new_state) {
  if (new_state == 0) { // Disconnected
    app_manager_set_state(APP_STATE_WIFI_CONNECTING);
  }
}

static const wifi_event_callbacks_t wifi_callbacks = {
  .on_connected = on_wifi_connected,
  .on_disconnected = on_wifi_disconnected,
  .on_error = on_wifi_error,
  .on_state_change = on_wifi_state_change,
};

naila_err_t app_manager_start(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  // Initialize watchdog for runtime protection
  esp_task_wdt_init(30, true);

  // Set initial state
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_INITIALIZING), TAG, "set state");

  // Start stats monitoring task
  // TODO: add watchdog task handle to the naila_stats_start_task function
  NAILA_PROPAGATE_ERROR(naila_stats_start_task(&stats_task_handle), TAG, "start stats task");

  NAILA_PROPAGATE_ERROR(
    // TODO: add watchdog task handle to the wifi_start_task function
      wifi_start_task(&wifi_callbacks, &wifi_task_handle), TAG, "start wifi task");

  if (wifi_task_handle) {
    esp_task_wdt_add(wifi_task_handle);
  }

  NAILA_LOGI(TAG, "Application started with modular task architecture");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

naila_err_t app_manager_stop(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_SHUTDOWN), TAG, "set shutdown state");

  // Stop modular tasks
  wifi_stop_task();
  naila_stats_stop_task();

  // Clean shutdown of subsystems
  wifi_cleanup();

  if (state_mutex) {
    vSemaphoreDelete(state_mutex);
    state_mutex = NULL;
  }

  if (stats_mutex) {
    vSemaphoreDelete(stats_mutex);
    stats_mutex = NULL;
  }

  // Update component state
  g_component_state = COMPONENT_STATE_UNINITIALIZED;

  NAILA_LOGI(TAG, "Application stopped cleanly");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

app_state_t app_manager_get_state(void) {
  app_state_t state;
  
  if (xSemaphoreTake(state_mutex, MUTEX_TIMEOUT) != pdTRUE) {
    NAILA_LOGE(TAG, "Failed to acquire state mutex for read");
    return APP_STATE_ERROR;  // Safe fallback
  }
  
  state = g_app_state;
  xSemaphoreGive(state_mutex);
  
  return state;
}

bool app_manager_is_running(void) {
  app_state_t state = app_manager_get_state();
  return state == APP_STATE_RUNNING;
}

naila_err_t app_manager_get_stats(app_stats_t *stats) {
  NAILA_CHECK_NULL(stats, TAG, "Stats pointer is null");
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  if (xSemaphoreTake(stats_mutex, MUTEX_TIMEOUT) != pdTRUE) {
    return NAILA_ERR_TIMEOUT;
  }

  // Copy existing stats
  memcpy(stats, &g_stats, sizeof(app_stats_t));
  xSemaphoreGive(stats_mutex);
  
  // Add runtime info
  stats->free_heap = esp_get_free_heap_size();
  stats->min_free_heap = esp_get_minimum_free_heap_size();
  
  return NAILA_OK;
}