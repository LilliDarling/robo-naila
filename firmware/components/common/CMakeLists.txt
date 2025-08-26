idf_component_register(
    SRCS "naila_log.c" "error_handling.c"
    INCLUDE_DIRS "include"
    REQUIRES "esp_timer" "app_trace"
)