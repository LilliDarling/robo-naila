#include <esp_event.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <esp_system.h>
#include <esp_timer.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <nvs_flash.h>

#include "app.h"
#include "common_types.h"
#include "config.h"
#include "error_handling.h"
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

static void on_wifi_connected(void) {
  NAILA_LOGI(TAG, "WiFi connection established");
}

static void on_wifi_disconnected(void) {
  NAILA_LOGW(TAG, "WiFi connection lost");
}

static void on_error(naila_err_t error) {
  NAILA_LOGE(TAG, "Application error occurred: %s", naila_err_to_string(error));
}

static const app_callbacks_t app_callbacks = {
  .on_state_change = on_state_change,
  .on_wifi_connected = on_wifi_connected,
  .on_wifi_disconnected = on_wifi_disconnected,
  .on_error = on_error,
};

static naila_err_t initialize_system(void) {
  // Initialize logging system first
  naila_log_init();

  // Log system info
  ESP_LOGI(TAG, "CPU freq: %d MHz", ets_get_cpu_frequency());
  ESP_LOGI(TAG, "Free heap: %d bytes", esp_get_free_heap_size());
  

  // Initialize NVS flash for system configuration
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    NAILA_ESP_CHECK(nvs_flash_erase(), TAG, "NVS flash erase");
    ret = nvs_flash_init();
  }
  NAILA_ESP_CHECK(ret, TAG, "NVS flash init");

  // Initialize network interface
  NAILA_ESP_CHECK(esp_netif_init(), TAG, "Network interface init");
  NAILA_ESP_CHECK(esp_event_loop_create_default(), TAG, "Event loop create");

  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

extern "C" void app_main() {
  NAILA_LOGI(TAG, "Starting NAILA Application...");
  NAILA_TIME_START(total_init);

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

  NAILA_TIME_END(TAG, total_init);
  NAILA_LOGI(TAG, "NAILA Robot Application started successfully");
}