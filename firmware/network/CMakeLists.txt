idf_component_register(
    SRCS "wifi.c" "naila_mqtt.c"
    INCLUDE_DIRS "include"
    REQUIRES "common" "config" "esp_wifi" "nvs_flash" "mqtt" "esp_event"
)
