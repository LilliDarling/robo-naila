# Get WiFi credentials from environment variables (same as main component)
if(DEFINED ENV{CONFIG_EXAMPLE_WIFI_SSID})
    set(WIFI_SSID $ENV{CONFIG_EXAMPLE_WIFI_SSID})
else()
    message(FATAL_ERROR "CONFIG_EXAMPLE_WIFI_SSID not set - run 'source setenv.sh' first")
endif()

if(DEFINED ENV{CONFIG_EXAMPLE_WIFI_PASSWORD})
    set(WIFI_PASSWORD $ENV{CONFIG_EXAMPLE_WIFI_PASSWORD})
else()
    # message(FATAL_ERROR "CONFIG_EXAMPLE_WIFI_PASSWORD not set - run 'source setenv.sh' first")
    set(WIFI_PASSWORD "")
endif()

idf_component_register(
    SRCS "config.c"
    INCLUDE_DIRS "include"
    REQUIRES "nvs_flash" "common"
)

# Set compile definitions for WiFi credentials
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    CONFIG_EXAMPLE_WIFI_SSID=\"${WIFI_SSID}\"
    CONFIG_EXAMPLE_WIFI_PASSWORD=\"${WIFI_PASSWORD}\"
)