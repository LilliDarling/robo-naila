#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>
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
} naila_wifi_config_t;

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

// Main system configuration
typedef struct {
  naila_wifi_config_t wifi;
  mqtt_config_t mqtt;
  ai_config_t ai;
} naila_config_t;

// Configuration manager API
// NOTE: Config is immutable after init. To change config, save to NVS then esp_restart().
naila_err_t config_manager_init(void);
const naila_config_t *config_manager_get(void);  // Returns pointer to immutable config

#ifdef __cplusplus
}
#endif

#endif