#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Standard error codes for NAILA components
typedef enum {
  NAILA_OK = ESP_OK,
  NAILA_FAIL = ESP_FAIL,
  NAILA_ERR_INVALID_ARG = ESP_ERR_INVALID_ARG,
  NAILA_ERR_NO_MEM = ESP_ERR_NO_MEM,
  NAILA_ERR_TIMEOUT = ESP_ERR_TIMEOUT,
  NAILA_ERR_NOT_INITIALIZED = 0x1000,
  NAILA_ERR_ALREADY_INITIALIZED = 0x1001,
  NAILA_ERR_WIFI_NOT_CONNECTED = 0x1002
} naila_err_t;

#ifdef __cplusplus
}
#endif

#endif