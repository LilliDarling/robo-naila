#include <esp_event.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <esp_clk_tree.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <nvs_flash.h>

#include "app.h"
#include "common_types.h"
#include "naila_log.h"


static const char *TAG = "main_app";

static const TickType_t SHUTDOWN_DELAY_MS = pdMS_TO_TICKS(1000);

static void restart_with_delay(const char* reason) {
  NAILA_LOGE(TAG, "Restarting due to: %s", reason);
  vTaskDelay(SHUTDOWN_DELAY_MS);  // Give time for logs to flush
  esp_restart();
}

// Application callback functions
static void on_state_change(app_state_t new_state) {
  NAILA_LOGI(TAG, "Application state changed to: %d", new_state);
}

static void on_error(naila_err_t error) {
  NAILA_LOGE(TAG, "Application error occurred: 0x%x", error);
}

static const app_callbacks_t app_callbacks = {
  .on_state_change = on_state_change,
  .on_error = on_error,
};

static naila_err_t initialize_system(void) {
  // Log system info
  uint32_t cpu_freq_mhz = 0;
  esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &cpu_freq_mhz);
  ESP_LOGI(TAG, "CPU freq: %lu MHz", cpu_freq_mhz / 1000000);
  ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());

  // Initialize NVS flash for system configuration
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ret = nvs_flash_erase();
    if (ret != ESP_OK) {
      NAILA_LOGE(TAG, "NVS flash erase failed: 0x%x", ret);
      return (naila_err_t)ret;
    }
    ret = nvs_flash_init();
  }
  if (ret != ESP_OK) {
    NAILA_LOGE(TAG, "NVS flash init failed: 0x%x", ret);
    return (naila_err_t)ret;
  }

  // Initialize network interface
  ret = esp_netif_init();
  if (ret != ESP_OK) {
    NAILA_LOGE(TAG, "Network interface init failed: 0x%x", ret);
    return (naila_err_t)ret;
  }

  ret = esp_event_loop_create_default();
  if (ret != ESP_OK) {
    NAILA_LOGE(TAG, "Event loop create failed: 0x%x", ret);
    return (naila_err_t)ret;
  }

  return NAILA_OK;
}

extern "C" void app_main() {
  NAILA_LOGI(TAG, "Starting NAILA Application...");

  // Initialize basic system components
  naila_err_t result = initialize_system();
  if (result != NAILA_OK) {
    restart_with_delay("System initialization failed");
  }

  // Initialize application manager
  result = app_manager_init(&app_callbacks);
  if (result != NAILA_OK) {
    restart_with_delay("Application manager initialization failed");
  }

  // Start application services
  result = app_manager_start();
  if (result != NAILA_OK) {
    restart_with_delay("Application start failed");
  }

  NAILA_LOGI(TAG, "NAILA Robot Application started successfully");
}
