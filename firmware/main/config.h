#ifndef CONFIG_H
#define CONFIG_H

// Include generated credentials headers
#include "wifi_credentials.h"
#include "mqtt_credentials.h"

// WiFi Configuration - these come from CMake compile definitions
#define CONFIG_WIFI_SSID CONFIG_EXAMPLE_WIFI_SSID
#define CONFIG_WIFI_PASSWORD CONFIG_EXAMPLE_WIFI_PASSWORD
#define CONFIG_WIFI_MAXIMUM_RETRY 5

// MQTT Configuration is now loaded from mqtt_credentials.h
// which is generated from environment variables

#endif