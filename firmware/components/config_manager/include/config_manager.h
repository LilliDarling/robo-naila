#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include "common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// WiFi Configuration
typedef struct {
  char ssid[32];
  char password[64];
  int max_retry;
  int timeout_ms;
} wifi_config_t;

// MQTT Configuration
typedef struct {
  char broker_ip[16];
  int broker_port;
  char robot_id[32];
  int keepalive_sec;
  int qos_level;
} mqtt_config_t;

// AI Configuration
typedef struct {
  char model_path[256];
  int tensor_arena_size;
  int inference_timeout_ms;
  bool enable_debug;
} ai_config_t;

// Audio Configuration
typedef struct {
  int sample_rate;
  int bit_depth;
  int channels;
  int buffer_size;
} audio_config_t;

// Main system configuration
typedef struct {
  wifi_config_t wifi;
  mqtt_config_t mqtt;
  ai_config_t ai;
  audio_config_t audio;
  component_info_t info;
} naila_config_t;

// Configuration manager API
naila_err_t config_manager_init(void);
naila_err_t config_manager_load_defaults(naila_config_t *config);
naila_err_t config_manager_load_from_nvs(naila_config_t *config);
naila_err_t config_manager_save_to_nvs(const naila_config_t *config);
naila_err_t config_manager_validate(const naila_config_t *config);
const naila_config_t *config_manager_get(void);
naila_err_t config_manager_update_wifi(const wifi_config_t *wifi_config);
naila_err_t config_manager_update_mqtt(const mqtt_config_t *mqtt_config);

#ifdef __cplusplus
}
#endif

#endif