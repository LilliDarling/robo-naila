#include "app_manager.h"
#include "config_manager.h"
#include "error_handling.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "naila_log.h"
#include "wifi_manager.h"
#include <string.h>

static const char *TAG = "app_manager";
static component_state_t g_component_state = COMPONENT_STATE_UNINITIALIZED;
static app_state_t g_app_state = APP_STATE_INITIALIZING;
static app_callbacks_t g_callbacks = {0};
static app_stats_t g_stats = {0};


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
  if (new_state == g_app_state) {
    return NAILA_OK;
  }

  app_state_t old_state = g_app_state;
  g_app_state = new_state;

  NAILA_LOGI(TAG, "State transition: %s -> %s", app_state_to_string(old_state),
      app_state_to_string(new_state));

  if (g_callbacks.on_state_change) {
    g_callbacks.on_state_change(old_state, new_state);
  }

  return NAILA_OK;
}

naila_err_t app_manager_init(const app_callbacks_t *callbacks) {
  NAILA_LOG_FUNC_ENTER(TAG);

  if (g_component_state == COMPONENT_STATE_INITIALIZED) {
    return NAILA_ERR_ALREADY_INITIALIZED;
  }

  g_component_state = COMPONENT_STATE_INITIALIZING;

  // Copy callbacks if provided
  if (callbacks) {
    memcpy(&g_callbacks, callbacks, sizeof(app_callbacks_t));
  }

  // Initialize configuration manager
  NAILA_PROPAGATE_ERROR(config_manager_init(), TAG, "config manager init");

  // Initialize WiFi manager
  NAILA_PROPAGATE_ERROR(wifi_manager_init(), TAG, "wifi manager init");

  // Initialize statistics
  memset(&g_stats, 0, sizeof(app_stats_t));

  g_component_state = COMPONENT_STATE_INITIALIZED;
  NAILA_LOGI(TAG, "Application manager initialized");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}


// WiFi event callbacks for the app_manager
static void on_wifi_connected(void) {
  app_manager_set_state(APP_STATE_RUNNING);
  
  if (g_callbacks.on_wifi_connected) {
    g_callbacks.on_wifi_connected();
  }
}

static void on_wifi_disconnected(void) {
  g_stats.wifi_reconnect_count++;
  
  if (g_callbacks.on_wifi_disconnected) {
    g_callbacks.on_wifi_disconnected();
  }
}

static void on_wifi_error(naila_err_t error) {
  g_stats.error_count++;
  app_manager_set_state(APP_STATE_ERROR);
  
  if (g_callbacks.on_error) {
    g_callbacks.on_error(error);
  }
}

static void on_wifi_state_change(int new_state) {
  if (new_state == 0) { // Disconnected
    app_manager_set_state(APP_STATE_WIFI_CONNECTING);
  }
}


naila_err_t app_manager_start(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  // Set initial state
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_INITIALIZING), TAG, "set state");

  // Start stats monitoring task
  NAILA_PROPAGATE_ERROR(naila_stats_start_task(), TAG, "start stats task");

  // Start WiFi management task with callbacks
  wifi_event_callbacks_t wifi_callbacks = {
    .on_connected = on_wifi_connected,
    .on_disconnected = on_wifi_disconnected,
    .on_error = on_wifi_error,
    .on_state_change = on_wifi_state_change
  };
  
  NAILA_PROPAGATE_ERROR(
      wifi_manager_start_task(&wifi_callbacks), TAG, "start wifi task");

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
  wifi_manager_stop_task();
  naila_stats_stop_task();

  // Clean shutdown of subsystems
  wifi_manager_deinit();

  // Update component state
  g_component_state = COMPONENT_STATE_UNINITIALIZED;

  NAILA_LOGI(TAG, "Application stopped cleanly");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

app_state_t app_manager_get_state(void) { return g_app_state; }

bool app_manager_is_running(void) { return g_app_state == APP_STATE_RUNNING; }

naila_err_t app_manager_get_stats(app_stats_t *stats) {
  NAILA_CHECK_NULL(stats, TAG, "Stats pointer is null");
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");


  memcpy(stats, &g_stats, sizeof(app_stats_t));
  return NAILA_OK;
}