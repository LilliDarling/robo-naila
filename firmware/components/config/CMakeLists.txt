# Get WiFi credentials from environment variables
if(DEFINED ENV{CONFIG_EXAMPLE_WIFI_SSID})
    set(WIFI_SSID $ENV{CONFIG_EXAMPLE_WIFI_SSID})
else()
    message(FATAL_ERROR "CONFIG_EXAMPLE_WIFI_SSID not set - run 'source setenv.sh' first")
endif()

if(DEFINED ENV{CONFIG_EXAMPLE_WIFI_PASSWORD})
    set(WIFI_PASSWORD $ENV{CONFIG_EXAMPLE_WIFI_PASSWORD})
else()
    set(WIFI_PASSWORD "")
endif()

# Get MQTT credentials from environment variables
if(DEFINED ENV{CONFIG_MQTT_BROKER_IP})
    set(MQTT_BROKER_IP $ENV{CONFIG_MQTT_BROKER_IP})
else()
    set(MQTT_BROKER_IP "10.0.0.117")
    message(WARNING "CONFIG_MQTT_BROKER_IP not set - using default: ${MQTT_BROKER_IP}")
endif()

if(DEFINED ENV{CONFIG_MQTT_BROKER_PORT})
    set(MQTT_BROKER_PORT $ENV{CONFIG_MQTT_BROKER_PORT})
else()
    set(MQTT_BROKER_PORT "1883")
    message(WARNING "CONFIG_MQTT_BROKER_PORT not set - using default: ${MQTT_BROKER_PORT}")
endif()

if(DEFINED ENV{CONFIG_ROBOT_ID})
    set(ROBOT_ID $ENV{CONFIG_ROBOT_ID})
else()
    set(ROBOT_ID "naila_robot_001")
    message(WARNING "CONFIG_ROBOT_ID not set - using default: ${ROBOT_ID}")
endif()

idf_component_register(
    SRCS "config.c"
    INCLUDE_DIRS "include"
    REQUIRES "nvs_flash" "common"
)

# Set compile definitions for all credentials
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    CONFIG_EXAMPLE_WIFI_SSID=\"${WIFI_SSID}\"
    CONFIG_EXAMPLE_WIFI_PASSWORD=\"${WIFI_PASSWORD}\"
    CONFIG_MQTT_BROKER_IP=\"${MQTT_BROKER_IP}\"
    CONFIG_MQTT_BROKER_PORT=${MQTT_BROKER_PORT}
    CONFIG_ROBOT_ID=\"${ROBOT_ID}\"
)