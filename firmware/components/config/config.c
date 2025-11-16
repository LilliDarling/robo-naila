#include "config.h"
#include "error_handling.h"
#include "naila_log.h"
#include "nvs.h"
#include "nvs_flash.h"
#include <string.h>

static const char *TAG = "CONFIG_MANAGER";
static const char *NVS_NAMESPACE = "naila_config";
static naila_config_t g_config;
static component_state_t g_state = COMPONENT_STATE_UNINITIALIZED;

naila_err_t config_manager_init(void) {
  NAILA_LOG_FUNC_ENTER(TAG);

  if (g_state == COMPONENT_STATE_INITIALIZED) {
    return NAILA_ERR_ALREADY_INITIALIZED;
  }

  g_state = COMPONENT_STATE_INITIALIZING;

  NAILA_PROPAGATE_ERROR(
      config_manager_load_defaults(&g_config), TAG, "load defaults");

  g_config.info.name = "config_manager";
  g_config.info.version = "0.1.0";
  g_config.info.state = COMPONENT_STATE_INITIALIZED;
  g_state = COMPONENT_STATE_INITIALIZED;

  NAILA_LOGI(TAG, "Configuration manager initialized successfully");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

naila_err_t config_manager_load_defaults(naila_config_t *config) {
  NAILA_CHECK_NULL(config, TAG, "Config pointer is null");

  // WiFi defaults
  strncpy(config->wifi.ssid, CONFIG_EXAMPLE_WIFI_SSID,
      sizeof(config->wifi.ssid) - 1);
  strncpy(config->wifi.password, CONFIG_EXAMPLE_WIFI_PASSWORD,
      sizeof(config->wifi.password) - 1);
  config->wifi.max_retry = 50;
  config->wifi.timeout_ms = 30000;

  // MQTT defaults
  strncpy(config->mqtt.broker_ip, "10.0.0.117",
      sizeof(config->mqtt.broker_ip) - 1);
  config->mqtt.broker_port = 1883;
  strncpy(config->mqtt.robot_id, "naila_robot_001",
      sizeof(config->mqtt.robot_id) - 1);
  config->mqtt.keepalive_sec = 60;
  config->mqtt.qos_level = 1;

  // AI defaults
  strncpy(config->ai.model_path, "/spiffs/model.tflite",
      sizeof(config->ai.model_path) - 1);
  config->ai.tensor_arena_size = 64 * 1024;
  config->ai.inference_timeout_ms = 5000;
  config->ai.enable_debug = false;

  // Audio defaults
  config->audio.sample_rate = 16000;
  config->audio.bit_depth = 16;
  config->audio.channels = 1;
  config->audio.buffer_size = 1024;

  return NAILA_OK;
}

naila_err_t config_manager_load_from_nvs(naila_config_t *config) {
  NAILA_CHECK_NULL(config, TAG, "Config pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "config_manager");

  nvs_handle_t nvs_handle;
  esp_err_t ret = nvs_open(NVS_NAMESPACE, NVS_READONLY, &nvs_handle);
  if (ret == ESP_ERR_NVS_NOT_FOUND) {
    NAILA_LOGI(TAG, "NVS namespace not found - this is normal on first boot");
    return NAILA_ERR_NOT_INITIALIZED;
  } else if (ret != ESP_OK) {
    NAILA_ESP_CHECK(ret, TAG, "NVS open");
  }

  size_t required_size = sizeof(naila_config_t);
  esp_err_t err = nvs_get_blob(nvs_handle, "config", config, &required_size);
  nvs_close(nvs_handle);

  if (err == ESP_ERR_NVS_NOT_FOUND) {
    NAILA_LOGI(TAG, "Configuration blob not found in NVS - using defaults");
    return NAILA_FAIL;
  }

  NAILA_ESP_CHECK(err, TAG, "NVS get blob");
  NAILA_LOGI(TAG, "Configuration loaded from NVS");
  return NAILA_OK;
}

naila_err_t config_manager_save_to_nvs(const naila_config_t *config) {
  NAILA_CHECK_NULL(config, TAG, "Config pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "config_manager");

  NAILA_PROPAGATE_ERROR(
      config_manager_validate(config), TAG, "config validation");

  nvs_handle_t nvs_handle;
  NAILA_ESP_CHECK(
      nvs_open(NVS_NAMESPACE, NVS_READWRITE, &nvs_handle), TAG, "NVS open");

  esp_err_t err =
      nvs_set_blob(nvs_handle, "config", config, sizeof(naila_config_t));
  if (err == ESP_OK) {
    err = nvs_commit(nvs_handle);
  }
  nvs_close(nvs_handle);

  NAILA_ESP_CHECK(err, TAG, "NVS save");
  NAILA_LOGI(TAG, "Configuration saved to NVS");
  return NAILA_OK;
}

naila_err_t config_manager_validate(const naila_config_t *config) {
  NAILA_CHECK_NULL(config, TAG, "Config pointer is null");

  // Validate WiFi config
  NAILA_CHECK(strlen(config->wifi.ssid) > 0, TAG, NAILA_ERR_INVALID_ARG,
      "WiFi SSID is empty");
  NAILA_CHECK(strlen(config->wifi.ssid) < 32, TAG, NAILA_ERR_INVALID_ARG,
      "WiFi SSID too long");
  NAILA_CHECK(config->wifi.max_retry > 0 && config->wifi.max_retry <= 10, TAG,
      NAILA_ERR_INVALID_ARG, "WiFi max_retry out of range");

  // Validate MQTT config
  NAILA_CHECK(strlen(config->mqtt.broker_ip) > 0, TAG, NAILA_ERR_INVALID_ARG,
      "MQTT broker IP is empty");
  NAILA_CHECK(config->mqtt.broker_port > 0 && config->mqtt.broker_port <= 65535,
      TAG, NAILA_ERR_INVALID_ARG, "MQTT port out of range");
  NAILA_CHECK(strlen(config->mqtt.robot_id) > 0, TAG, NAILA_ERR_INVALID_ARG,
      "Robot ID is empty");

  // Validate AI config
  NAILA_CHECK(config->ai.tensor_arena_size >= 1024, TAG, NAILA_ERR_INVALID_ARG,
      "Tensor arena size too small");
  NAILA_CHECK(config->ai.inference_timeout_ms > 0, TAG, NAILA_ERR_INVALID_ARG,
      "Invalid inference timeout");

  // Validate audio config
  NAILA_CHECK(config->audio.sample_rate > 0, TAG, NAILA_ERR_INVALID_ARG,
      "Invalid sample rate");
  NAILA_CHECK(config->audio.channels > 0 && config->audio.channels <= 2, TAG,
      NAILA_ERR_INVALID_ARG, "Invalid channel count");

  return NAILA_OK;
}

const naila_config_t *config_manager_get(void) {
  if (g_state != COMPONENT_STATE_INITIALIZED) {
    NAILA_LOGE(TAG, "Config manager not initialized");
    return NULL;
  }
  return &g_config;
}

naila_err_t config_manager_update_wifi(const naila_wifi_config_t *wifi_config) {
  NAILA_CHECK_NULL(wifi_config, TAG, "WiFi config pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "config_manager");

  memcpy(&g_config.wifi, wifi_config, sizeof(naila_wifi_config_t));
  NAILA_LOGI(TAG, "WiFi configuration updated");
  return config_manager_save_to_nvs(&g_config);
}

naila_err_t config_manager_update_mqtt(const mqtt_config_t *mqtt_config) {
  NAILA_CHECK_NULL(mqtt_config, TAG, "MQTT config pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "config_manager");

  memcpy(&g_config.mqtt, mqtt_config, sizeof(mqtt_config_t));
  NAILA_LOGI(TAG, "MQTT configuration updated");
  return config_manager_save_to_nvs(&g_config);
}