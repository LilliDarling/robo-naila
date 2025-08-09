idf_component_register(
    SRCS "app_manager.c" "app_manager.c"
    INCLUDE_DIRS "include"
    REQUIRES "common" "config_manager" "wifi" "esp_timer"
)