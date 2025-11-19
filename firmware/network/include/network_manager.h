#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "common_types.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Network events
typedef enum {
  NETWORK_EVENT_CONTROL_PLANE_READY,
  NETWORK_EVENT_WIFI_CONNECTED,
  NETWORK_EVENT_WIFI_DISCONNECTED,
  NETWORK_EVENT_MQTT_CONNECTED,
  NETWORK_EVENT_MQTT_DISCONNECTED,
  NETWORK_EVENT_ERROR
} network_event_t;

// Network event callback
typedef void (*network_event_callback_t)(network_event_t event);

// Network configuration
typedef struct {
  network_event_callback_t callback;
} network_config_t;

// Network manager API
naila_err_t network_manager_init(network_config_t* config);
naila_err_t network_manager_start(void);
naila_err_t network_manager_stop(void);
bool network_manager_is_ready(void);
bool network_manager_is_wifi_connected(void);
bool network_manager_is_mqtt_connected(void);

#ifdef __cplusplus
}
#endif

#endif // NETWORK_MANAGER_H
