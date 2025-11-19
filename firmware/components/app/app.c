#include "app.h"
#include "config.h"
#include "error_handling.h"
#include "mutex_utils.h"
#include "esp_system.h"
#include "esp_timer.h"
#include <esp_task_wdt.h>
#include <freertos/semphr.h>

#include "network_manager.h"
#include "naila_log.h"
#include <string.h>

static TaskHandle_t stats_task_handle = NULL;

static SemaphoreHandle_t state_mutex = NULL;
static SemaphoreHandle_t stats_mutex = NULL;

static const char *TAG = "APP_MANAGER";
static const app_callbacks_t *g_callbacks = NULL;

static component_state_t g_component_state = COMPONENT_STATE_UNINITIALIZED;
static app_state_t g_app_state = APP_STATE_INITIALIZING;
static app_stats_t g_stats = {0};

// Forward declaration
static void on_network_event(network_event_t event);


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
  app_state_t old_state;

  NAILA_PROPAGATE_ERROR(mutex_execute(state_mutex, ^(void* ctx) {
    old_state = g_app_state;
    g_app_state = new_state;
    return NAILA_OK;
  }, NULL), TAG, "set state");

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
    vSemaphoreDelete(state_mutex);
    state_mutex = NULL;
    return NAILA_ERR_NO_MEMORY;
  }

  g_component_state = COMPONENT_STATE_INITIALIZING;

  g_callbacks = callbacks;

  // Initialize configuration manager
  naila_err_t result = config_manager_init();
  if (result != NAILA_OK) {
    vSemaphoreDelete(state_mutex);
    vSemaphoreDelete(stats_mutex);
    state_mutex = NULL;
    stats_mutex = NULL;
    NAILA_LOG_ERROR(TAG, result, "Error in config manager init: config_manager_init()");
    return result;
  }

  // Initialize network manager (WiFi + MQTT)
  network_config_t network_config = {
    .callback = on_network_event
  };
  result = network_manager_init(&network_config);
  if (result != NAILA_OK) {
    vSemaphoreDelete(state_mutex);
    vSemaphoreDelete(stats_mutex);
    state_mutex = NULL;
    stats_mutex = NULL;
    NAILA_LOG_ERROR(TAG, result, "Error in network manager init: network_manager_init()");
    return result;
  }

  // Initialize statistics
  memset(&g_stats, 0, sizeof(app_stats_t));

  g_component_state = COMPONENT_STATE_INITIALIZED;
  NAILA_LOGI(TAG, "Application manager initialized");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

// Network event callback for the app_manager
static void on_network_event(network_event_t event) {
  switch (event) {
    case NETWORK_EVENT_WIFI_CONNECTED:
      NAILA_LOGI(TAG, "Network event: WiFi connected");
      app_manager_set_state(APP_STATE_SERVICES_STARTING);
      if (g_callbacks && g_callbacks->on_wifi_connected) {
        g_callbacks->on_wifi_connected();
      }
      break;

    case NETWORK_EVENT_MQTT_CONNECTED:
      NAILA_LOGI(TAG, "Network event: MQTT connected");
      break;

    case NETWORK_EVENT_CONTROL_PLANE_READY:
      NAILA_LOGI(TAG, "Network event: Control plane ready (WiFi + MQTT)");
      app_manager_set_state(APP_STATE_RUNNING);
      break;

    case NETWORK_EVENT_WIFI_DISCONNECTED:
      NAILA_LOGI(TAG, "Network event: WiFi disconnected");
      mutex_execute(stats_mutex, ^(void* ctx) {
        g_stats.wifi_reconnect_count++;
        return NAILA_OK;
      }, NULL);
      app_manager_set_state(APP_STATE_WIFI_CONNECTING);
      if (g_callbacks && g_callbacks->on_wifi_disconnected) {
        g_callbacks->on_wifi_disconnected();
      }
      break;

    case NETWORK_EVENT_MQTT_DISCONNECTED:
      NAILA_LOGI(TAG, "Network event: MQTT disconnected");
      break;

    case NETWORK_EVENT_ERROR:
      NAILA_LOGE(TAG, "Network event: Error");
      mutex_execute(stats_mutex, ^(void* ctx) {
        g_stats.error_count++;
        return NAILA_OK;
      }, NULL);
      app_manager_set_state(APP_STATE_ERROR);
      if (g_callbacks && g_callbacks->on_error) {
        g_callbacks->on_error(NAILA_FAIL);
      }
      break;

    default:
      NAILA_LOGW(TAG, "Unknown network event: %d", event);
      break;
  }
}

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

  // Start network manager (WiFi + MQTT initialization)
  NAILA_PROPAGATE_ERROR(network_manager_start(), TAG, "start network manager");

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
  network_manager_stop();
  naila_stats_stop_task();

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

  naila_err_t result = mutex_execute(state_mutex, ^(void* ctx) {
    state = g_app_state;
    return NAILA_OK;
  }, NULL);

  if (result != NAILA_OK) {
    NAILA_LOGE(TAG, "Failed to acquire state mutex for read");
    return APP_STATE_ERROR;
  }

  return state;
}

bool app_manager_is_running(void) {
  app_state_t state = app_manager_get_state();
  return state == APP_STATE_RUNNING;
}

naila_err_t app_manager_get_stats(app_stats_t *stats) {
  NAILA_CHECK_NULL(stats, TAG, "Stats pointer is null");
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  NAILA_PROPAGATE_ERROR(mutex_execute(stats_mutex, ^(void* ctx) {
    memcpy(stats, &g_stats, sizeof(app_stats_t));
    return NAILA_OK;
  }, NULL), TAG, "get stats");

  // Add runtime info
  stats->free_heap = esp_get_free_heap_size();
  stats->min_free_heap = esp_get_minimum_free_heap_size();

  return NAILA_OK;
}