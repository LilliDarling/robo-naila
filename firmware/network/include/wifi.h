#ifndef WIFI_H
#define WIFI_H

#include "common_types.h"
#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *ssid;
  const char *password;
  int max_retry;
} wifi_config_simple_t;

naila_err_t wifi_init(void);
naila_err_t wifi_connect(const wifi_config_simple_t *config);
bool wifi_is_connected(void);
naila_err_t wifi_disconnect(void);

// Task management functions
typedef struct {
  void (*on_connected)(void);
  void (*on_error)(naila_err_t error);
} wifi_event_callbacks_t;

naila_err_t wifi_start_task(const wifi_event_callbacks_t *callbacks);
naila_err_t wifi_stop_task(void);
bool wifi_is_task_running(void);

#ifdef __cplusplus
}
#endif

#endif