#ifndef APP_MANAGER_H
#define APP_MANAGER_H

#include "common_types.h"

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
  void (*on_state_change)(app_state_t old_state, app_state_t new_state);
  void (*on_wifi_connected)(void);
  void (*on_wifi_disconnected)(void);
  void (*on_error)(naila_err_t error);
} app_callbacks_t;

// Application statistics
typedef struct {
  uint32_t uptime_sec;
  uint32_t wifi_reconnect_count;
  uint32_t inference_count;
  uint32_t error_count;
  size_t free_heap_bytes;
  size_t min_free_heap_bytes;
} app_stats_t;

// Application manager API
naila_err_t app_manager_init(const app_callbacks_t *callbacks);
naila_err_t app_manager_start(void);
naila_err_t app_manager_stop(void);
app_state_t app_manager_get_state(void);
naila_err_t app_manager_get_stats(app_stats_t *stats);
naila_err_t app_manager_run_main_loop(void);

// State management
naila_err_t app_manager_set_state(app_state_t new_state);
bool app_manager_is_running(void);

#endif