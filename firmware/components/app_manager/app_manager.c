#include "app_manager.h"
#include "config_manager.h"
#include "error_handling.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "wifi_manager.h"
#include "naila_log.h"
#include <string.h>

static const char *TAG = "app_manager";
static component_state_t g_component_state = COMPONENT_STATE_UNINITIALIZED;
static app_state_t g_app_state = APP_STATE_INITIALIZING;
static app_callbacks_t g_callbacks = {0};
static app_stats_t g_stats = {0};
static int64_t g_start_time = 0;

// FreeRTOS coordination primitives
static EventGroupHandle_t app_events = NULL;
static TaskHandle_t wifi_task_handle = NULL;
static TaskHandle_t stats_task_handle = NULL;

// Event bits
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_READY_BIT BIT1
#define SHUTDOWN_BIT BIT2

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

  // Create event group
  app_events = xEventGroupCreate();
  if (app_events == NULL) {
    NAILA_LOGE(TAG, "Failed to create event group");
    return NAILA_ERR_MEMORY_ALLOCATION;
  }

  // Initialize statistics
  memset(&g_stats, 0, sizeof(app_stats_t));
  g_stats.min_free_heap_bytes = esp_get_free_heap_size();

  g_component_state = COMPONENT_STATE_INITIALIZED;
  NAILA_LOGI(TAG, "Application manager initialized");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}


static void wifi_management_task(void *parameters) {
  NAILA_LOGI(TAG, "WiFi management task started");

  // Initialize WiFi manager once
  esp_err_t wifi_init_result = wifi_manager_init();
  if (wifi_init_result != ESP_OK) {
    NAILA_LOG_ERROR(TAG, wifi_init_result, "WiFi manager initialization failed");
    app_manager_set_state(APP_STATE_ERROR);
    if (g_callbacks.on_error) {
      g_callbacks.on_error(NAILA_ERR_WIFI_NOT_CONNECTED);
    }
    wifi_task_handle = NULL;
    vTaskDelete(NULL);
    return;
  }

  const naila_config_t *config = config_manager_get();
  if (config == NULL) {
    NAILA_LOGE(TAG, "Failed to get configuration");
    app_manager_set_state(APP_STATE_ERROR);
    wifi_task_handle = NULL;
    vTaskDelete(NULL);
    return;
  }

  wifi_config_simple_t wifi_config = {
    .ssid = config->wifi.ssid,
    .password = config->wifi.password,
    .max_retry = config->wifi.max_retry
  };

  bool first_connection = true;

  while (true) {
    // Check for shutdown signal
    EventBits_t bits = xEventGroupWaitBits(app_events, SHUTDOWN_BIT, pdFALSE, pdFALSE, 0);
    if (bits & SHUTDOWN_BIT) {
      NAILA_LOGI(TAG, "WiFi management task shutting down");
      break;
    }

    // Check WiFi connection status
    if (!wifi_manager_is_connected()) {
      if (!first_connection) {
        NAILA_LOGE(TAG, "WiFi disconnected during runtime");
        g_stats.wifi_reconnect_count++;
        if (g_callbacks.on_wifi_disconnected) {
          g_callbacks.on_wifi_disconnected();
        }
        wifi_manager_reset_connection_state();
      }

      // Set state to connecting
      app_manager_set_state(APP_STATE_WIFI_CONNECTING);
      NAILA_LOGI(TAG, "Connecting to WiFi SSID: %s", wifi_config.ssid);

      // Attempt WiFi connection
      esp_err_t wifi_connect_result = wifi_manager_connect(&wifi_config);
      if (wifi_connect_result == ESP_OK) {
        NAILA_LOGI(TAG, "WiFi connection successful");
        
        // Set both connected and ready bits for MQTT coordination
        xEventGroupSetBits(app_events, WIFI_CONNECTED_BIT | WIFI_READY_BIT);
        
        // Set application state to running
        app_manager_set_state(APP_STATE_RUNNING);
        
        if (g_callbacks.on_wifi_connected) {
          g_callbacks.on_wifi_connected();
        }
        
        first_connection = false;
      } else {
        NAILA_LOG_ERROR(TAG, wifi_connect_result, "WiFi connection failed");
        g_stats.error_count++;
        
        if (first_connection) {
          // On initial connection failure, set error state
          app_manager_set_state(APP_STATE_ERROR);
          if (g_callbacks.on_error) {
            g_callbacks.on_error(NAILA_ERR_WIFI_NOT_CONNECTED);
          }
        }
      }
    }

    // Sleep for WiFi check interval
    vTaskDelay(pdMS_TO_TICKS(5000)); // 5 seconds
  }

  // Clean up task handle
  wifi_task_handle = NULL;
  vTaskDelete(NULL);
}

static void stats_monitoring_task(void *parameters) {
  NAILA_LOGI(TAG, "Stats monitoring task started");

  while (true) {
    // Check for shutdown signal
    EventBits_t bits = xEventGroupWaitBits(app_events, SHUTDOWN_BIT, pdFALSE, pdFALSE, 0);
    if (bits & SHUTDOWN_BIT) {
      NAILA_LOGI(TAG, "Stats monitoring task shutting down");
      break;
    }

    // Update statistics
    g_stats.uptime_sec = (esp_timer_get_time() - g_start_time) / 1000000;
    size_t current_free_heap = esp_get_free_heap_size();
    g_stats.free_heap_bytes = current_free_heap;
    if (current_free_heap < g_stats.min_free_heap_bytes) {
      g_stats.min_free_heap_bytes = current_free_heap;
    }

    // Log status periodically
    if (g_stats.uptime_sec % 360 == 0 && g_stats.uptime_sec > 0) { // Every 5 min
      NAILA_LOGI(TAG, "System status - Uptime: %lu sec, Free heap: %zu bytes",
          g_stats.uptime_sec, g_stats.free_heap_bytes);
    }

    // Sleep for stats update interval
    vTaskDelay(pdMS_TO_TICKS(1000)); // 1 second
  }

  // Clean up task handle
  stats_task_handle = NULL;
  vTaskDelete(NULL);
}

naila_err_t app_manager_start(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  // State: Initializing
  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_INITIALIZING), TAG, "set state");

  // Create WiFi management task
  BaseType_t wifi_task_result = xTaskCreate(
      wifi_management_task,
      "wifi_mgmt",
      4096,
      NULL,
      5,
      &wifi_task_handle);
  
  if (wifi_task_result != pdPASS) {
    NAILA_LOGE(TAG, "Failed to create WiFi management task");
    return NAILA_ERR_NO_MEM;
  }

  // Create stats monitoring task
  BaseType_t stats_task_result = xTaskCreate(
      stats_monitoring_task,
      "stats_mon",
      3072,
      NULL,
      2,
      &stats_task_handle);
  
  if (stats_task_result != pdPASS) {
    NAILA_LOGE(TAG, "Failed to create stats monitoring task");
    return NAILA_ERR_NO_MEM;
  }

  NAILA_LOGI(TAG, "Application started successfully with task-based architecture");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}


naila_err_t app_manager_stop(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_component_state, TAG, "app_manager");

  NAILA_PROPAGATE_ERROR(
      app_manager_set_state(APP_STATE_SHUTDOWN), TAG, "set shutdown state");

  // Signal shutdown to all tasks
  if (app_events != NULL) {
    xEventGroupSetBits(app_events, SHUTDOWN_BIT);
  }

  // Wait for tasks to clean up
  vTaskDelay(pdMS_TO_TICKS(1000));

  // Clean shutdown of subsystems
  NAILA_LOGI(TAG, "Performing graceful WiFi shutdown");
  wifi_manager_deinit();

  // Clean up event group
  if (app_events != NULL) {
    vEventGroupDelete(app_events);
    app_events = NULL;
  }

  // Update component state
  g_component_state = COMPONENT_STATE_UNINITIALIZED;

  NAILA_LOGI(TAG, "Application stopped with graceful cleanup");
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