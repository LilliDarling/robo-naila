idf_component_register(
    SRCS "app.c"
    INCLUDE_DIRS "include"
    REQUIRES "common" "config" "wifi" "esp_timer"
)