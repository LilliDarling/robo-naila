#ifndef APP_H
#define APP_H

#include "common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Application states
typedef enum {
  APP_STATE_INITIALIZING,
  APP_STATE_WIFI_CONNECTING,
  APP_STATE_SERVICES_STARTING,
  APP_STATE_RUNNING,
  APP_STATE_ERROR,
  APP_STATE_SHUTDOWN
} app_state_t;

// Application manager callbacks
typedef struct {
  void (*on_state_change)(app_state_t new_state);
  void (*on_wifi_connected)(void);
  void (*on_wifi_disconnected)(void);
  void (*on_error)(naila_err_t error);
} app_callbacks_t;

// Application manager API
naila_err_t app_manager_init(const app_callbacks_t *callbacks);
naila_err_t app_manager_start(void);
naila_err_t app_manager_stop(void);
app_state_t app_manager_get_state(void);
bool app_manager_is_running(void);

#ifdef __cplusplus
}
#endif

#endif