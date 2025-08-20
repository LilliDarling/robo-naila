#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

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

naila_err_t wifi_manager_init(void);
naila_err_t wifi_manager_connect(const wifi_config_simple_t *config);
bool wifi_manager_is_connected(void);
naila_err_t wifi_manager_disconnect(void);
// ADDED: Functions for graceful WiFi cleanup and state reset - can be removed if causing issues
naila_err_t wifi_manager_deinit(void);
naila_err_t wifi_manager_reset_connection_state(void);
// END ADDED
naila_err_t wifi_manager_get_info(component_info_t *info);

// Task management functions
typedef struct {
  void (*on_connected)(void);
  void (*on_disconnected)(void);
  void (*on_error)(naila_err_t error);
  void (*on_state_change)(int new_state);
} wifi_event_callbacks_t;

naila_err_t wifi_manager_start_task(const wifi_event_callbacks_t *callbacks);
naila_err_t wifi_manager_stop_task(void);
bool wifi_manager_is_task_running(void);

#ifdef __cplusplus
}
#endif

#endif