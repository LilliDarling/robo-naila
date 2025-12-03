#include "config.h"
#include "naila_log.h"
#include "nvs.h"
#include "nvs_flash.h"
#include <string.h>

static const char *TAG = "CONFIG_MANAGER";
static const char *NVS_NAMESPACE = "naila_config";
static naila_config_t g_config;

// Forward declarations
static void config_manager_load_defaults(naila_config_t *config);
static naila_err_t config_manager_load_from_nvs(naila_config_t *config);

naila_err_t config_manager_init(void) {
  config_manager_load_defaults(&g_config);

  naila_err_t nvs_result = config_manager_load_from_nvs(&g_config);
  if (nvs_result == NAILA_OK) {
    NAILA_LOGI(TAG, "Loaded configuration from NVS");
  } else {
    NAILA_LOGI(TAG, "Using default configuration");
  }

  NAILA_LOGI(TAG, "Config manager initialized");
  return NAILA_OK;
}

static void config_manager_load_defaults(naila_config_t *config) {
  // Zero entire struct - ensures all strings are null-terminated
  memset(config, 0, sizeof(*config));

  // WiFi defaults
  strncpy(config->wifi.ssid, CONFIG_EXAMPLE_WIFI_SSID, sizeof(config->wifi.ssid) - 1);
  strncpy(config->wifi.password, CONFIG_EXAMPLE_WIFI_PASSWORD, sizeof(config->wifi.password) - 1);
  config->wifi.max_retry = 50;
  config->wifi.timeout_ms = 30000;

  // MQTT defaults
  strncpy(config->mqtt.broker_ip, CONFIG_MQTT_BROKER_IP, sizeof(config->mqtt.broker_ip) - 1);
  config->mqtt.broker_port = CONFIG_MQTT_BROKER_PORT;
  strncpy(config->mqtt.robot_id, CONFIG_ROBOT_ID, sizeof(config->mqtt.robot_id) - 1);
  config->mqtt.keepalive_sec = 60;
  config->mqtt.qos_level = 1;

  // AI defaults
  strncpy(config->ai.model_path, "/spiffs/model.tflite", sizeof(config->ai.model_path) - 1);
  config->ai.tensor_arena_size = 64 * 1024;
  config->ai.inference_timeout_ms = 5000;
  config->ai.enable_debug = false;
}

static naila_err_t config_manager_load_from_nvs(naila_config_t *config) {
  nvs_handle_t nvs_handle;
  esp_err_t ret = nvs_open(NVS_NAMESPACE, NVS_READONLY, &nvs_handle);
  if (ret == ESP_ERR_NVS_NOT_FOUND) {
    NAILA_LOGI(TAG, "NVS namespace not found - this is normal on first boot");
    return NAILA_ERR_NOT_INITIALIZED;
  } else if (ret != ESP_OK) {
    NAILA_LOGE(TAG, "NVS open failed: 0x%x", ret);
    return (naila_err_t)ret;
  }

  size_t required_size = sizeof(naila_config_t);
  esp_err_t err = nvs_get_blob(nvs_handle, "config", config, &required_size);
  nvs_close(nvs_handle);

  if (err == ESP_ERR_NVS_NOT_FOUND) {
    NAILA_LOGI(TAG, "Configuration blob not found in NVS - using defaults");
    return NAILA_FAIL;
  }

  if (err != ESP_OK) {
    NAILA_LOGE(TAG, "NVS get blob failed: 0x%x", err);
    return (naila_err_t)err;
  }

  NAILA_LOGI(TAG, "Configuration loaded from NVS");
  return NAILA_OK;
}

const naila_config_t *config_manager_get(void) {
  return &g_config;
}