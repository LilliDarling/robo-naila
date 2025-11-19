idf_component_register(
    SRCS "app.c"
    INCLUDE_DIRS "include"
    REQUIRES "common" "config" "network" "esp_timer"
)