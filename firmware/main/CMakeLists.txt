
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

idf_component_register(SRCS "main.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES "common" "app_manager" "config_manager"
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

# Add binary directory to include path so the generated header can be found
target_include_directories(${COMPONENT_LIB} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Set compile definitions for WiFi credentials
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    CONFIG_EXAMPLE_WIFI_SSID=\"${WIFI_SSID}\"
)