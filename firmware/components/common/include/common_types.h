#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include "esp_err.h"
#include "esp_log.h"

// Standard error codes for NAILA components
typedef enum {
  NAILA_OK = ESP_OK,
  NAILA_FAIL = ESP_FAIL,
  NAILA_ERR_INVALID_ARG = ESP_ERR_INVALID_ARG,
  NAILA_ERR_NO_MEM = ESP_ERR_NO_MEM,
  NAILA_ERR_TIMEOUT = ESP_ERR_TIMEOUT,
  NAILA_ERR_NOT_INITIALIZED = 0x1000,
  NAILA_ERR_ALREADY_INITIALIZED = 0x1001,
  NAILA_ERR_WIFI_NOT_CONNECTED = 0x1002,
  NAILA_ERR_AI_MODEL_LOAD_FAILED = 0x1003,
  NAILA_ERR_AUDIO_INIT_FAILED = 0x1004,
  NAILA_ERR_INFERENCE_FAILED = 0x1005
} naila_err_t;

// Component initialization states
typedef enum {
  COMPONENT_STATE_UNINITIALIZED,
  COMPONENT_STATE_INITIALIZING,
  COMPONENT_STATE_INITIALIZED,
  COMPONENT_STATE_ERROR
} component_state_t;

// Standard component info structure
typedef struct {
  const char *name;
  const char *version;
  component_state_t state;
} component_info_t;

#endif