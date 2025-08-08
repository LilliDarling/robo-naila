#include "app_manager.h"
#include "config_manager.h"
#include "error_handling.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "wifi_manager.h"
#include "naila_log.h"
#include <string.h>

static const char *TAG = "app_manager";
static component_state_t g_component_state = COMPONENT_STATE_UNINITIALIZED;
static app_state_t g_app_state = APP_STATE_INITIALIZING;
static app_callbacks_t g_callbacks = {0};
static app_stats_t g_stats = {0};
static int64_t g_start_time = 0;

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
  g_start_time = esp_timer_get_time();

  // Copy callbacks if provided
  if (callbacks) {
    memcpy(&g_callbacks, callbacks, sizeof(app_callbacks_t));
  }

  // Initialize configuration manager
  NAILA_PROPAGATE_ERROR(config_manager_init(), TAG, "config manager init");

  // Initialize statistics
  memset(&g_stats, 0, sizeof(app_stats_t));
  g_stats.min_free_heap_bytes = esp_get_free_heap_size();

  g_component_state = COMPONENT_STATE_INITIALIZED;
  NAILA_LOGI(TAG, "Application manager initialized");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

static naila_err_t initialize_wifi_subsystem(void) {
  NAILA_LOG_FUNC_ENTER(TAG);

  const naila_config_t *config = config_manager_get();
  NAILA_CHECK_NULL(config, TAG, "Failed to get configuration");

  NAILA_LOGI(TAG, "Initializing WiFi subsystem...");

  // Initialize WiFi manager
  esp_err_t wifi_init_result = wifi_manager_init();
  if (wifi_init_result != ESP_OK) {
    NAILA_LOG_ERROR(
        TAG, wifi_init_result, "WiFi manager initialization failed");
    return NAILA_ERR_WIFI_NOT_CONNECTED;
  }

  // Connect to WiFi
  wifi_config_simple_t wifi_config = {.ssid = config->wifi.ssid,
      .password = config->wifi.password,
      .max_retry = config->wifi.max_retry};

  NAILA_LOGI(TAG, "Connecting to WiFi SSID: %s", wifi_config.ssid);
  esp_err_t wifi_connect_result = wifi_manager_connect(&wifi_config);
  if (wifi_connect_result != ESP_OK) {
    NAILA_LOG_ERROR(TAG, wifi_connect_result, "WiFi connection failed");
    g_stats.error_count++;
    return NAILA_ERR_WIFI_NOT_CONNECTED;
  }

  if (g_callbacks.on_wifi_connected) {
    g_callbacks.on_wifi_connected();
  }

  NAILA_LOGI(TAG, "WiFi subsystem initialized successfully");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

naila_err_t app_manager_start(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  // State: Initializing
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_INITIALIZING), TAG, "set state");

  // State: WiFi Connecting
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_WIFI_CONNECTING), TAG, "set state");
  naila_err_t wifi_result = initialize_wifi_subsystem();
  if (wifi_result != NAILA_OK) {
    NAILA_PROPAGATE_ERROR(
        app_manager_set_state(APP_STATE_ERROR), TAG, "set error state");
    if (g_callbacks.on_error) {
      g_callbacks.on_error(wifi_result);
    }
    return wifi_result;
  }

  // State: Services Starting
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_SERVICES_STARTING), TAG, "set state");

  // TODO: Initialize MQTT, AI inference, audio services here

  // State: Running
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_RUNNING), TAG, "set state");

  NAILA_LOGI(TAG, "Application started successfully");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

naila_err_t app_manager_run_main_loop(void) {
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  NAILA_LOGI(TAG, "Starting main application loop");

  while (app_manager_is_running()) {
    // Update statistics
    g_stats.uptime_sec = (esp_timer_get_time() - g_start_time) / 1000000;
    size_t current_free_heap = esp_get_free_heap_size();
    g_stats.free_heap_bytes = current_free_heap;
    if (current_free_heap < g_stats.min_free_heap_bytes) {
      g_stats.min_free_heap_bytes = current_free_heap;
    }

    // Check WiFi connection status
    if (!wifi_manager_is_connected()) {
      NAILA_LOGE(TAG, "WiFi disconnected during runtime");
      g_stats.wifi_reconnect_count++;
      if (g_callbacks.on_wifi_disconnected) {
        g_callbacks.on_wifi_disconnected();
      }

      // ADDED: Enhanced reconnection logic with state reset - can be removed if causing issues
      // Attempt to reconnect with state reset for better reliability
      NAILA_LOGW(TAG, "Attempting WiFi reconnection with state reset...");
      wifi_manager_reset_connection_state();
      
      // Re-initialize WiFi subsystem
      naila_err_t reconnect_result = initialize_wifi_subsystem();
      if (reconnect_result == NAILA_OK) {
        NAILA_LOGI(TAG, "WiFi reconnection successful");
      } else {
        NAILA_LOGE(TAG, "WiFi reconnection failed");
        g_stats.error_count++;
      }
      // END ADDED
    }

    // Main application processing would go here
    // For now, just log status periodically
    if (g_stats.uptime_sec % 30 == 0) { // Every 30 seconds
      NAILA_LOGI(TAG, "System status - Uptime: %lu sec, Free heap: %zu bytes",
          g_stats.uptime_sec, g_stats.free_heap_bytes);
    }

    // Sleep for main loop interval
    vTaskDelay(pdMS_TO_TICKS(1000)); // 1 second
  }

  NAILA_LOGI(TAG, "Main application loop ended");
  return NAILA_OK;
}

naila_err_t app_manager_stop(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_SHUTDOWN), TAG, "set shutdown state");

  // ADDED: Enhanced shutdown with graceful WiFi cleanup - can be removed if causing issues
  // Clean shutdown of subsystems - use deinit for proper cleanup
  NAILA_LOGI(TAG, "Performing graceful WiFi shutdown");
  wifi_manager_deinit();

  // Update component state
  g_component_state = COMPONENT_STATE_UNINITIALIZED;

  NAILA_LOGI(TAG, "Application stopped with graceful cleanup");
  // END ADDED
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

app_state_t app_manager_get_state(void) { return g_app_state; }

bool app_manager_is_running(void) { return g_app_state == APP_STATE_RUNNING; }

naila_err_t app_manager_get_stats(app_stats_t *stats) {
  NAILA_CHECK_NULL(stats, TAG, "Stats pointer is null");
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  // Update current stats before returning
  g_stats.uptime_sec = (esp_timer_get_time() - g_start_time) / 1000000;
  g_stats.free_heap_bytes = esp_get_free_heap_size();

  memcpy(stats, &g_stats, sizeof(app_stats_t));
  return NAILA_OK;
}