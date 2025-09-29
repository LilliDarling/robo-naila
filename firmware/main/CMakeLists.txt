
# Get WiFi credentials from environment variables (set by setenv.sh)
if(DEFINED ENV{CONFIG_EXAMPLE_WIFI_SSID})
    set(WIFI_SSID $ENV{CONFIG_EXAMPLE_WIFI_SSID})
else()
    message(FATAL_ERROR "CONFIG_EXAMPLE_WIFI_SSID not set - run 'source setenv.sh' first")
endif()

if(DEFINED ENV{CONFIG_EXAMPLE_WIFI_PASSWORD})
    set(WIFI_PASSWORD $ENV{CONFIG_EXAMPLE_WIFI_PASSWORD})
else()
    set(WIFI_PASSWORD "")
    # message(FATAL_ERROR "CONFIG_EXAMPLE_WIFI_PASSWORD not set - run 'source setenv.sh' first")
endif()

# Get MQTT credentials from environment variables (set by setenv.sh)
if(DEFINED ENV{CONFIG_MQTT_BROKER_IP})
    set(MQTT_BROKER_IP $ENV{CONFIG_MQTT_BROKER_IP})
else()
    set(MQTT_BROKER_IP "192.168.1.100")
    message(WARNING "CONFIG_MQTT_BROKER_IP not set - using default: 192.168.1.100")
endif()

if(DEFINED ENV{CONFIG_MQTT_BROKER_PORT})
    set(MQTT_BROKER_PORT $ENV{CONFIG_MQTT_BROKER_PORT})
else()
    set(MQTT_BROKER_PORT "1883")
    message(WARNING "CONFIG_MQTT_BROKER_PORT not set - using default: 1883")
endif()

if(DEFINED ENV{CONFIG_ROBOT_ID})
    set(ROBOT_ID $ENV{CONFIG_ROBOT_ID})
else()
    set(ROBOT_ID "naila_robot_001")
    message(WARNING "CONFIG_ROBOT_ID not set - using default: naila_robot_001")
endif()

idf_component_register(SRCS "main.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES "common" "app" "config"
                       PRIV_REQUIRES "esp_wifi"
                                     "nvs_flash"
                                     "esp_netif"
                                     "esp_event"
                                     "esp_timer"
                                     "wifi"
                      )

# Generate WiFi credentials header file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/wifi_credentials.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wifi_credentials.h"
    @ONLY
)

# Generate MQTT credentials header file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mqtt_credentials.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mqtt_credentials.h"
    @ONLY
)

# Add binary directory to include path so the generated header can be found
target_include_directories(${COMPONENT_LIB} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Set compile definitions for WiFi credentials
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    CONFIG_EXAMPLE_WIFI_SSID=\"${WIFI_SSID}\"
)