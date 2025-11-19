idf_component_register(
    SRCS "wifi.c" "network_manager.c" "mqtt_client.c"
    INCLUDE_DIRS "include"
    REQUIRES "common" "config" "esp_wifi" "nvs_flash" "mqtt" "esp_event"
)
