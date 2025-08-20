#include "naila_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

static const char *STATS_TAG = "naila_stats";

// Stats monitoring task state
static TaskHandle_t stats_task_handle = NULL;
static bool stats_task_should_stop = false;
static system_stats_t g_stats = {0};
static int64_t g_start_time = 0;

void naila_log_init(void) {
#if NAILA_USE_APP_TRACE
  // Initialize app trace for high-performance logging
  esp_apptrace_init();
#endif
  
  // Set production log levels for efficiency
#if NAILA_LOGGING_ENABLED && !NAILA_USE_APP_TRACE
  // Reduce default log levels for production
  esp_log_level_set("*", ESP_LOG_WARN);  // Default to warnings only
  esp_log_level_set("wifi", ESP_LOG_ERROR); // WiFi errors only
  esp_log_level_set("esp_netif_lwip", ESP_LOG_ERROR);
  esp_log_level_set("tcpip_adapter", ESP_LOG_ERROR);
#endif

  // Initialize stats
  g_start_time = esp_timer_get_time();
  memset(&g_stats, 0, sizeof(system_stats_t));
  g_stats.min_free_heap_bytes = esp_get_free_heap_size();
}

// Stats monitoring task implementation
static void stats_monitoring_task(void *parameters) {
  while (!stats_task_should_stop) {
    // Update statistics
    g_stats.uptime_sec = (esp_timer_get_time() - g_start_time) / 1000000;
    size_t current_free_heap = esp_get_free_heap_size();
    g_stats.free_heap_bytes = current_free_heap;
    if (current_free_heap < g_stats.min_free_heap_bytes) {
      g_stats.min_free_heap_bytes = current_free_heap;
    }

    // Log status periodically (every 10 minutes)
    if (g_stats.uptime_sec % 600 == 0 && g_stats.uptime_sec > 0) {
      NAILA_LOGI(STATS_TAG, "System status - Uptime: %lu sec, Free heap: %zu bytes",
          g_stats.uptime_sec, g_stats.free_heap_bytes);
    }

    vTaskDelay(pdMS_TO_TICKS(1000)); // Update every second
  }

  stats_task_handle = NULL;
  vTaskDelete(NULL);
}

naila_err_t naila_stats_start_task(void) {
  if (stats_task_handle != NULL) {
    return NAILA_ERR_ALREADY_INITIALIZED;
  }
  
  stats_task_should_stop = false;
  
  BaseType_t result = xTaskCreate(
      stats_monitoring_task,
      "stats_mon",
      3072,
      NULL,
      2,
      &stats_task_handle);
  
  return (result == pdPASS) ? NAILA_OK : NAILA_ERR_NO_MEM;
}

naila_err_t naila_stats_stop_task(void) {
  if (stats_task_handle == NULL) {
    return NAILA_OK;
  }
  
  stats_task_should_stop = true;
  vTaskDelay(pdMS_TO_TICKS(1000)); // Wait for task to exit
  
  if (stats_task_handle != NULL) {
    vTaskDelete(stats_task_handle);
    stats_task_handle = NULL;
  }
  
  return NAILA_OK;
}

naila_err_t naila_stats_get(system_stats_t *stats) {
  if (stats == NULL) {
    return NAILA_ERR_INVALID_ARG;
  }
  
  // Update current stats before returning
  g_stats.uptime_sec = (esp_timer_get_time() - g_start_time) / 1000000;
  g_stats.free_heap_bytes = esp_get_free_heap_size();
  
  memcpy(stats, &g_stats, sizeof(system_stats_t));
  return NAILA_OK;
}

bool naila_stats_is_task_running(void) {
  return stats_task_handle != NULL;
}