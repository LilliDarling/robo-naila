#ifndef CONFIG_H
#define CONFIG_H

// Include generated WiFi credentials header
#include "wifi_credentials.h"

// WiFi Configuration - these come from CMake compile definitions
#define CONFIG_WIFI_SSID CONFIG_EXAMPLE_WIFI_SSID
#define CONFIG_WIFI_PASSWORD CONFIG_EXAMPLE_WIFI_PASSWORD
#define CONFIG_WIFI_MAXIMUM_RETRY 5

// MQTT Configuration
#define CONFIG_MQTT_BROKER_IP "192.168.1.100" // Replace with your server IP
#define CONFIG_MQTT_BROKER_PORT 1883
#define CONFIG_ROBOT_ID "naila_robot_001"

#endif