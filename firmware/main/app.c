#include "app.h"
#include "config.h"
#include "esp_system.h"
#include "esp_timer.h"
#include <esp_task_wdt.h>
#include <freertos/semphr.h>

#include "network_manager.h"
#include "naila_log.h"

static const char *TAG = "APP_MANAGER";

// Configuration constants
#define WATCHDOG_TIMEOUT_SEC 30
#define WATCHDOG_PANIC_ON_TIMEOUT true

// Unified application manager state
typedef struct {
    bool initialized;
    app_state_t state;
    const app_callbacks_t *callbacks;
} app_manager_t;

static app_manager_t g_app = {0};
static SemaphoreHandle_t g_app_mutex = NULL;

// Forward declaration
static void on_network_event(network_event_t event);

naila_err_t app_manager_init(const app_callbacks_t *callbacks) {
  // Create mutex on first init (never delete it)
  if (!g_app_mutex) {
    g_app_mutex = xSemaphoreCreateMutex();
    if (!g_app_mutex) {
      NAILA_LOGE(TAG, "Failed to create app mutex");
      return NAILA_ERR_NO_MEM;
    }
  }

  if (xSemaphoreTake(g_app_mutex, portMAX_DELAY) != pdTRUE) {
    NAILA_LOGE(TAG, "Failed to acquire mutex in init");
    return NAILA_FAIL;
  }

  if (g_app.initialized) {
    xSemaphoreGive(g_app_mutex);
    NAILA_LOGW(TAG, "Application manager already initialized");
    return NAILA_ERR_ALREADY_INITIALIZED;
  }

  g_app.initialized = true;
  g_app.callbacks = callbacks;
  g_app.state = APP_STATE_INITIALIZING;
  xSemaphoreGive(g_app_mutex);

  // Initialize configuration manager
  naila_err_t result = config_manager_init();
  if (result != NAILA_OK) {
    if (xSemaphoreTake(g_app_mutex, portMAX_DELAY) == pdTRUE) {
      g_app.initialized = false;
      xSemaphoreGive(g_app_mutex);
    } else {
      NAILA_LOGW(TAG, "Failed to acquire mutex during config init rollback");
    }
    NAILA_LOG_ERROR(TAG, result, "Error in config manager init: config_manager_init()");
    return result;
  }

  // Initialize network manager (WiFi + MQTT)
  network_config_t network_config = {
    .callback = on_network_event
  };
  result = network_manager_init(&network_config);
  if (result != NAILA_OK) {
    // Rollback: reset initialized flag (config manager has no cleanup needed)
    if (xSemaphoreTake(g_app_mutex, portMAX_DELAY) == pdTRUE) {
      g_app.initialized = false;
      xSemaphoreGive(g_app_mutex);
    } else {
      NAILA_LOGW(TAG, "Failed to acquire mutex during network init rollback");
    }
    NAILA_LOG_ERROR(TAG, result, "Error in network manager init: network_manager_init()");
    return result;
  }

  NAILA_LOGI(TAG, "Application manager initialized");
  return NAILA_OK;
}

// Network event callback for the app_manager
static void on_network_event(network_event_t event) {
  app_state_t new_state = APP_STATE_ERROR;
  const app_callbacks_t *callbacks = NULL;
  bool should_call_state_change = false;
  bool should_call_wifi_connected = false;
  bool should_call_wifi_disconnected = false;
  bool should_call_error = false;

  // Acquire mutex once and perform all state updates
  if (!g_app_mutex || xSemaphoreTake(g_app_mutex, portMAX_DELAY) != pdTRUE) {
    NAILA_LOGE(TAG, "Failed to acquire mutex in network event callback");
    return;
  }

  callbacks = g_app.callbacks;

  switch (event) {
    case NETWORK_EVENT_WIFI_CONNECTED:
      NAILA_LOGI(TAG, "Network event: WiFi connected");
      new_state = APP_STATE_SERVICES_STARTING;
      g_app.state = new_state;
      should_call_state_change = true;
      should_call_wifi_connected = true;
      break;

    case NETWORK_EVENT_CONTROL_PLANE_READY:
      NAILA_LOGI(TAG, "Network event: Control plane ready (WiFi + MQTT)");
      new_state = APP_STATE_RUNNING;
      g_app.state = new_state;
      should_call_state_change = true;
      break;

    case NETWORK_EVENT_WIFI_DISCONNECTED:
      NAILA_LOGI(TAG, "Network event: WiFi disconnected");
      new_state = APP_STATE_WIFI_CONNECTING;
      g_app.state = new_state;
      should_call_state_change = true;
      should_call_wifi_disconnected = true;
      break;

    case NETWORK_EVENT_ERROR:
      NAILA_LOGE(TAG, "Network event: Error");
      new_state = APP_STATE_ERROR;
      g_app.state = new_state;
      should_call_state_change = true;
      should_call_error = true;
      break;

    case NETWORK_EVENT_MQTT_CONNECTED:
    case NETWORK_EVENT_MQTT_DISCONNECTED:
      // These events are logged but don't trigger state changes or callbacks
      // MQTT state is already reflected in CONTROL_PLANE_READY event
      break;

    default:
      NAILA_LOGW(TAG, "Unknown network event: %d", event);
      break;
  }

  xSemaphoreGive(g_app_mutex);

  // Call callbacks outside of mutex to prevent deadlocks
  if (callbacks) {
    if (should_call_state_change && callbacks->on_state_change) {
      callbacks->on_state_change(new_state);
    }
    if (should_call_wifi_connected && callbacks->on_wifi_connected) {
      callbacks->on_wifi_connected();
    }
    if (should_call_wifi_disconnected && callbacks->on_wifi_disconnected) {
      callbacks->on_wifi_disconnected();
    }
    if (should_call_error && callbacks->on_error) {
      callbacks->on_error(NAILA_FAIL);
    }
  }
}

naila_err_t app_manager_start(void) {
  bool is_initialized = false;
  if (!g_app_mutex || xSemaphoreTake(g_app_mutex, portMAX_DELAY) != pdTRUE) {
    NAILA_LOGE(TAG, "Failed to acquire mutex in start");
    return NAILA_FAIL;
  }

  is_initialized = g_app.initialized;
  xSemaphoreGive(g_app_mutex);

  if (!is_initialized) {
    NAILA_LOGE(TAG, "Application manager not initialized");
    return NAILA_ERR_NOT_INITIALIZED;
  }

  // Initialize watchdog for runtime protection
  esp_err_t wdt_result = esp_task_wdt_init(WATCHDOG_TIMEOUT_SEC, WATCHDOG_PANIC_ON_TIMEOUT);
  if (wdt_result != ESP_OK) {
    NAILA_LOGE(TAG, "Failed to initialize watchdog: 0x%x", wdt_result);
    return (naila_err_t)wdt_result;
  }

  // Start network manager (WiFi + MQTT initialization)
  naila_err_t err = network_manager_start();
  if (err != NAILA_OK) {
    NAILA_LOGE(TAG, "Failed to start network manager: 0x%x", err);
    return err;
  }

  NAILA_LOGI(TAG, "Application started");
  return NAILA_OK;
}

naila_err_t app_manager_stop(void) {
  if (!g_app_mutex) {
    return NAILA_OK;
  }

  if (xSemaphoreTake(g_app_mutex, portMAX_DELAY) != pdTRUE) {
    NAILA_LOGE(TAG, "Failed to acquire mutex in stop");
    return NAILA_FAIL;
  }

  if (!g_app.initialized) {
    xSemaphoreGive(g_app_mutex);
    return NAILA_OK;
  }

  g_app.initialized = false;
  g_app.callbacks = NULL;
  g_app.state = APP_STATE_SHUTDOWN;
  xSemaphoreGive(g_app_mutex);

  // Stop network manager
  network_manager_stop();

  NAILA_LOGI(TAG, "Application stopped");
  return NAILA_OK;
}

app_state_t app_manager_get_state(void) {
  app_state_t state = APP_STATE_ERROR;

  if (g_app_mutex && xSemaphoreTake(g_app_mutex, portMAX_DELAY) == pdTRUE) {
    state = g_app.state;
    xSemaphoreGive(g_app_mutex);
  }

  return state;
}

bool app_manager_is_running(void) {
  bool running = false;

  if (g_app_mutex && xSemaphoreTake(g_app_mutex, portMAX_DELAY) == pdTRUE) {
    running = (g_app.state == APP_STATE_RUNNING);
    xSemaphoreGive(g_app_mutex);
  }

  return running;
}