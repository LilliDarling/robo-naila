#ifndef NAILA_MQTT_H
#define NAILA_MQTT_H

#include "esp_err.h"
#include "config.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*naila_mqtt_message_handler_t)(const char* topic, const char* data, int data_len);

esp_err_t naila_mqtt_init(const mqtt_config_t* config);
esp_err_t naila_mqtt_publish(const char* topic, const char* data, int len, int qos);
esp_err_t naila_mqtt_subscribe(const char* topic, int qos);
void naila_mqtt_register_handler(naila_mqtt_message_handler_t handler);
bool naila_mqtt_is_connected(void);

#ifdef __cplusplus
}
#endif

#endif // NAILA_MQTT_H