idf_component_register(SRCS "app.c" "main.cpp"
                       REQUIRES "common" "config" "network"
                       PRIV_REQUIRES "nvs_flash"
                                     "esp_netif"
                                     "esp_timer"
                      )