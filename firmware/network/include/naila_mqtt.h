#ifndef NAILA_MQTT_H
#define NAILA_MQTT_H

#include "common_types.h"
#include "config.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mqtt_message_handler_t)(const char* topic, const char* data, int data_len);

naila_err_t mqtt_client_init(const mqtt_config_t* config);
naila_err_t mqtt_client_stop(void);
naila_err_t mqtt_client_publish(const char* topic, const char* data, int len, int qos);
naila_err_t mqtt_client_subscribe(const char* topic, int qos);
void mqtt_client_register_handler(mqtt_message_handler_t handler);
bool mqtt_client_is_connected(void);

#ifdef __cplusplus
}
#endif

#endif // NAILA_MQTT_H
