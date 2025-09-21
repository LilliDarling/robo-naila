idf_component_register(
    SRCS "naila_mqtt.c"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES mqtt esp_event
)