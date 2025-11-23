#include "wifi.h"
#include "config.h"
#include "naila_log.h"
#include "mutex_helpers.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>

static const char *TAG = "WIFI";

#define WIFI_CONNECT_DELAY_MS 1000
#define WIFI_DISCONNECT_DELAY_MS 500
#define WIFI_RECONNECT_TIMEOUT_MS 10000
#define WIFI_RECONNECT_RETRY_DELAY_MS 5000
#define WIFI_TASK_STACK_SIZE 2048
#define WIFI_TASK_PRIORITY 5
#define WIFI_TASK_STOP_WAIT_ITERATIONS 10
#define WIFI_TASK_STOP_WAIT_DELAY_MS 100

typedef struct {
    TaskHandle_t task_handle;
    wifi_event_callbacks_t callbacks;
    bool initialized;
    bool task_should_stop;
    bool connected;
} wifi_state_t;

static wifi_state_t g_wifi = {0};

static SemaphoreHandle_t wifi_mutex = NULL;

// Forward declarations
static void wifi_reconnection_task(void *pvParameters);
static naila_err_t start_reconnection_task(void);
static const wifi_disconnect_reason_t *find_disconnect_reason(wifi_err_reason_t reason);

static void event_handler(void *arg __attribute__((unused)),
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data) {

  if (event_base == WIFI_EVENT) {
    switch (event_id) {
      case WIFI_EVENT_STA_CONNECTED:
        NAILA_LOGI(TAG, "Connected successfully");
        break;

      case WIFI_EVENT_STA_DISCONNECTED: {
        wifi_event_sta_disconnected_t *disconnected_event =
            (wifi_event_sta_disconnected_t *)event_data;

        const wifi_disconnect_reason_t *reason_info =
            find_disconnect_reason(disconnected_event->reason);
        if (reason_info) {
          if (reason_info->log_level == ESP_LOG_ERROR) {
            NAILA_LOGE(TAG, "%s", reason_info->message);
          } else {
            NAILA_LOGI(TAG, "%s", reason_info->message);
          }
        } else {
          NAILA_LOGE(TAG, "Unknown disconnect reason: %d", disconnected_event->reason);
        }

        bool should_start_task = false;
        MUTEX_LOCK_VOID(wifi_mutex, TAG) {
          g_wifi.connected = false;
          should_start_task = !g_wifi.task_handle;
        } MUTEX_UNLOCK_VOID();

        if (should_start_task) {
          naila_err_t result = start_reconnection_task();
          if (result != NAILA_OK) {
            NAILA_LOGE(TAG, "Failed to start reconnection task: 0x%x", result);
          }
        }
        break;
      }

      default:
        break;
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    wifi_event_callbacks_t callbacks = {0};

    MUTEX_LOCK_VOID(wifi_mutex, TAG) {
      g_wifi.connected = true;
      if (g_wifi.task_handle) {
        g_wifi.task_should_stop = true;
      }
      callbacks = g_wifi.callbacks;
    } MUTEX_UNLOCK_VOID();

    if (callbacks.on_connected) {
      callbacks.on_connected();
    }
  }
}

naila_err_t wifi_init(void) {
  if (!wifi_mutex) {
    wifi_mutex = xSemaphoreCreateMutex();
    if (!wifi_mutex) {
      NAILA_LOGE(TAG, "Failed to create WiFi mutex");
      return NAILA_ERR_NO_MEM;
    }
  }

  MUTEX_LOCK(wifi_mutex, TAG) {
    if (g_wifi.initialized) {
      xSemaphoreGive(wifi_mutex);
      NAILA_LOGW(TAG, "WiFi already initialized");
      return NAILA_ERR_ALREADY_INITIALIZED;
    }
    g_wifi.initialized = true;
  } MUTEX_UNLOCK();

  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_err_t err = esp_wifi_init(&cfg);
  if (err != ESP_OK) {
    MUTEX_LOCK(wifi_mutex, TAG) {
      g_wifi.initialized = false;
    } MUTEX_UNLOCK();
    NAILA_LOGE(TAG, "WiFi init failed: %s", esp_err_to_name(err));
    return (naila_err_t)err;
  }

  esp_wifi_set_country_code("US", true);

  err = esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL);
  if (err != ESP_OK) {
    NAILA_LOGE(TAG, "Failed to register WiFi event handler: 0x%x", err);
  }

  err = esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL);
  if (err != ESP_OK) {
    NAILA_LOGE(TAG, "Failed to register IP event handler: 0x%x", err);
  }

  NAILA_LOGI(TAG, "WiFi initialized");
  return NAILA_OK;
}

naila_err_t wifi_connect(const wifi_config_simple_t *config) {
  MUTEX_LOCK(wifi_mutex, TAG) {
    if (!g_wifi.initialized) {
      xSemaphoreGive(wifi_mutex);
      NAILA_LOGE(TAG, "WiFi not initialized");
      return NAILA_ERR_NOT_INITIALIZED;
    }
  } MUTEX_UNLOCK();

  wifi_config_t wifi_config = {
      .sta = {
          .threshold.authmode = WIFI_AUTH_WPA2_PSK,
          .pmf_cfg = {.capable = true, .required = false},
      },
  };

  strlcpy((char *)wifi_config.sta.ssid, config->ssid, sizeof(wifi_config.sta.ssid));
  strlcpy((char *)wifi_config.sta.password, config->password, sizeof(wifi_config.sta.password));

  NAILA_LOGI(TAG, "Connecting to SSID: %s", wifi_config.sta.ssid);

  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_mode(WIFI_MODE_STA);
  esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
  esp_wifi_start();

  vTaskDelay(pdMS_TO_TICKS(WIFI_CONNECT_DELAY_MS));
  esp_err_t err = esp_wifi_connect();

  if (err != ESP_OK) {
    NAILA_LOGE(TAG, "Failed to initiate WiFi connection: %s", esp_err_to_name(err));
    return (naila_err_t)err;
  }

  // Return immediately - connection status will come via callback
  return NAILA_OK;
}

bool wifi_is_connected(void) {
  bool connected = false;
  if (wifi_mutex) {
    MUTEX_LOCK_BOOL(wifi_mutex, TAG) {
      connected = g_wifi.connected;
    } MUTEX_UNLOCK_BOOL();
  }
  return connected;
}

naila_err_t wifi_disconnect(void) {
  MUTEX_LOCK(wifi_mutex, TAG) {
    if (!g_wifi.initialized) {
      xSemaphoreGive(wifi_mutex);
      NAILA_LOGE(TAG, "WiFi not initialized");
      return NAILA_ERR_NOT_INITIALIZED;
    }
    g_wifi.connected = false;
  } MUTEX_UNLOCK();

  esp_err_t err = esp_wifi_disconnect();
  if (err != ESP_OK) {
    NAILA_LOGW(TAG, "WiFi disconnect failed: 0x%x", err);
  } else {
    vTaskDelay(pdMS_TO_TICKS(WIFI_DISCONNECT_DELAY_MS));
  }

  esp_wifi_stop();

  return NAILA_OK;
}

static naila_err_t start_reconnection_task(void) {
  bool already_running = false;
  TaskHandle_t new_task_handle = NULL;

  MUTEX_LOCK(wifi_mutex, TAG) {
    if (g_wifi.task_handle) {
      already_running = true;
    } else {
      g_wifi.task_should_stop = false;
    }
  } MUTEX_UNLOCK();

  if (already_running) {
    NAILA_LOGD(TAG, "Reconnection task already running, skipping");
    return NAILA_OK;
  }

  NAILA_LOGD(TAG, "Creating WiFi reconnection task");
  BaseType_t result = xTaskCreate(wifi_reconnection_task, "wifi_reconnect",
      WIFI_TASK_STACK_SIZE, NULL, WIFI_TASK_PRIORITY, &new_task_handle);

  if (result != pdPASS) {
    NAILA_LOGE(TAG, "Failed to create reconnection task");
    return NAILA_ERR_NO_MEM;
  }

  MUTEX_LOCK(wifi_mutex, TAG) {
    g_wifi.task_handle = new_task_handle;
  } MUTEX_UNLOCK();

  return NAILA_OK;
}

static void wifi_reconnection_task(void *pvParameters __attribute__((unused))) {
  const naila_config_t *config = config_manager_get();
  if (!config) {
    NAILA_LOGE(TAG, "Config unavailable");

    wifi_event_callbacks_t callbacks = {0};
    MUTEX_LOCK_VOID(wifi_mutex, TAG) {
      callbacks = g_wifi.callbacks;
      g_wifi.task_handle = NULL;
    } MUTEX_UNLOCK_VOID();

    if (callbacks.on_error) {
      callbacks.on_error(NAILA_ERR_INVALID_ARG);
    }
    vTaskDelete(NULL);
    return;
  }

  int attempt = 0;
  bool should_stop = false;

  while (!should_stop && !wifi_is_connected() && attempt < config->wifi.max_retry) {
    MUTEX_LOCK_VOID(wifi_mutex, TAG) {
      should_stop = g_wifi.task_should_stop;
    } MUTEX_UNLOCK_VOID();

    if (should_stop) break;

    attempt++;
    NAILA_LOGD(TAG, "WiFi connection attempt %d/%d", attempt, config->wifi.max_retry);

    esp_err_t err = esp_wifi_connect();
    if (err == ESP_OK) {
      // Poll for connection with timeout
      int elapsed_ms = 0;
      while (elapsed_ms < WIFI_RECONNECT_TIMEOUT_MS && !wifi_is_connected()) {
        vTaskDelay(pdMS_TO_TICKS(100));
        elapsed_ms += 100;

        // Check if we should stop
        MUTEX_LOCK_VOID(wifi_mutex, TAG) {
          should_stop = g_wifi.task_should_stop;
        } MUTEX_UNLOCK_VOID();
        if (should_stop) break;
      }

      if (wifi_is_connected()) {
        NAILA_LOGI(TAG, "WiFi reconnection successful");
        break;
      }
    } else {
      NAILA_LOGE(TAG, "Failed to initiate connection: %s", esp_err_to_name(err));
    }

    if (!should_stop) {
      vTaskDelay(pdMS_TO_TICKS(WIFI_RECONNECT_RETRY_DELAY_MS));
    }
  }

  wifi_event_callbacks_t callbacks = {0};
  MUTEX_LOCK_VOID(wifi_mutex, TAG) {
    callbacks = g_wifi.callbacks;
    g_wifi.task_handle = NULL;
  } MUTEX_UNLOCK_VOID();

  if (attempt >= config->wifi.max_retry && callbacks.on_error) {
    NAILA_LOGE(TAG, "Max reconnection attempts reached");
    callbacks.on_error(NAILA_ERR_WIFI_NOT_CONNECTED);
  }

  vTaskDelete(NULL);
}

naila_err_t wifi_start_task(const wifi_event_callbacks_t *callbacks) {
  MUTEX_LOCK(wifi_mutex, TAG) {
    if (!g_wifi.initialized) {
      xSemaphoreGive(wifi_mutex);
      NAILA_LOGE(TAG, "WiFi not initialized");
      return NAILA_ERR_NOT_INITIALIZED;
    }
    if (callbacks) {
      g_wifi.callbacks = *callbacks;
    }
  } MUTEX_UNLOCK();

  const naila_config_t *config = config_manager_get();
  if (config && !wifi_is_connected()) {
    wifi_config_simple_t wifi_cfg = {
      .ssid = config->wifi.ssid,
      .password = config->wifi.password,
      .max_retry = config->wifi.max_retry
    };

    naila_err_t result = wifi_connect(&wifi_cfg);
    if (result != NAILA_OK && !wifi_is_connected()) {
      NAILA_LOGI(TAG, "WiFi task started (reconnection mode)");
      return start_reconnection_task();
    }
  }

  NAILA_LOGI(TAG, "WiFi task started");
  return NAILA_OK;
}

naila_err_t wifi_stop_task(void) {
  TaskHandle_t task_to_stop = NULL;

  MUTEX_LOCK(wifi_mutex, TAG) {
    if (!g_wifi.task_handle) {
      xSemaphoreGive(wifi_mutex);
      return NAILA_OK;
    }
    g_wifi.task_should_stop = true;
    task_to_stop = g_wifi.task_handle;
  } MUTEX_UNLOCK();

  int wait_count = 0;
  bool task_still_running = true;
  while (task_still_running && wait_count < WIFI_TASK_STOP_WAIT_ITERATIONS) {
    vTaskDelay(pdMS_TO_TICKS(WIFI_TASK_STOP_WAIT_DELAY_MS));
    MUTEX_LOCK(wifi_mutex, TAG) {
      task_still_running = g_wifi.task_handle;
    } MUTEX_UNLOCK();
    wait_count++;
  }

  if (task_still_running && task_to_stop) {
    NAILA_LOGW(TAG, "Force deleting reconnection task");
    vTaskDelete(task_to_stop);
    MUTEX_LOCK(wifi_mutex, TAG) {
      g_wifi.task_handle = NULL;
    } MUTEX_UNLOCK();
  }

  return NAILA_OK;
}

bool wifi_is_task_running(void) {
  bool running = false;
  if (wifi_mutex) {
    MUTEX_LOCK_BOOL(wifi_mutex, TAG) {
      running = g_wifi.task_handle;
    } MUTEX_UNLOCK_BOOL();
  }
  return running;
}

// WiFi disconnect reason lookup table and helper
typedef struct {
  wifi_err_reason_t reason;
  esp_log_level_t log_level;
  const char *message;
} wifi_disconnect_reason_t;

static const wifi_disconnect_reason_t disconnect_reasons[] = {
    {WIFI_REASON_AUTH_EXPIRE, ESP_LOG_ERROR, "Authentication expired - auth timeout"},
    {WIFI_REASON_AUTH_LEAVE, ESP_LOG_ERROR, "Authentication leave - deauth from AP"},
    {WIFI_REASON_ASSOC_EXPIRE, ESP_LOG_ERROR, "Association expired - assoc timeout"},
    {WIFI_REASON_ASSOC_TOOMANY, ESP_LOG_ERROR, "Too many associations - AP at capacity"},
    {WIFI_REASON_NOT_AUTHED, ESP_LOG_ERROR, "Not authenticated - auth required first"},
    {WIFI_REASON_NOT_ASSOCED, ESP_LOG_ERROR, "Not associated - assoc required first"},
    {WIFI_REASON_ASSOC_LEAVE, ESP_LOG_ERROR, "Association leave - disassoc from AP"},
    {WIFI_REASON_ASSOC_NOT_AUTHED, ESP_LOG_ERROR, "Association not authenticated"},
    {WIFI_REASON_DISASSOC_PWRCAP_BAD, ESP_LOG_ERROR, "Disassoc due to bad power capability"},
    {WIFI_REASON_DISASSOC_SUPCHAN_BAD, ESP_LOG_ERROR, "Disassoc due to bad supported channels"},
    {WIFI_REASON_BSS_TRANSITION_DISASSOC, ESP_LOG_ERROR, "BSS transition disassociation"},
    {WIFI_REASON_IE_INVALID, ESP_LOG_ERROR, "Invalid information element"},
    {WIFI_REASON_MIC_FAILURE, ESP_LOG_ERROR, "MIC failure - encryption issue"},
    {WIFI_REASON_4WAY_HANDSHAKE_TIMEOUT, ESP_LOG_ERROR, "4-way handshake timeout - WPA issue"},
    {WIFI_REASON_GROUP_KEY_UPDATE_TIMEOUT, ESP_LOG_ERROR, "Group key update timeout"},
    {WIFI_REASON_IE_IN_4WAY_DIFFERS, ESP_LOG_ERROR, "IE in 4-way handshake differs"},
    {WIFI_REASON_GROUP_CIPHER_INVALID, ESP_LOG_ERROR, "Invalid group cipher"},
    {WIFI_REASON_PAIRWISE_CIPHER_INVALID, ESP_LOG_ERROR, "Invalid pairwise cipher"},
    {WIFI_REASON_AKMP_INVALID, ESP_LOG_ERROR, "Invalid AKMP (auth/key mgmt)"},
    {WIFI_REASON_UNSUPP_RSN_IE_VERSION, ESP_LOG_ERROR, "Unsupported RSN IE version"},
    {WIFI_REASON_INVALID_RSN_IE_CAP, ESP_LOG_ERROR, "Invalid RSN IE capabilities"},
    {WIFI_REASON_802_1X_AUTH_FAILED, ESP_LOG_ERROR, "802.1X authentication failed"},
    {WIFI_REASON_CIPHER_SUITE_REJECTED, ESP_LOG_ERROR, "Cipher suite rejected"},
    {WIFI_REASON_BEACON_TIMEOUT, ESP_LOG_ERROR, "Beacon timeout - lost connection to AP"},
    {WIFI_REASON_NO_AP_FOUND, ESP_LOG_ERROR, "No AP found - SSID not visible"},
    {WIFI_REASON_AUTH_FAIL, ESP_LOG_ERROR, "Authentication failed - wrong credentials"},
    {WIFI_REASON_ASSOC_FAIL, ESP_LOG_ERROR, "Association failed - AP rejected association"},
    {WIFI_REASON_HANDSHAKE_TIMEOUT, ESP_LOG_ERROR, "Handshake timeout - general timeout"},
    {WIFI_REASON_CONNECTION_FAIL, ESP_LOG_ERROR, "Connection failed - general failure"},
    {WIFI_REASON_AP_TSF_RESET, ESP_LOG_ERROR, "AP TSF reset"},
    {WIFI_REASON_ROAMING, ESP_LOG_INFO, "Roaming to another AP"}
};

static const size_t disconnect_reasons_count = sizeof(disconnect_reasons) / sizeof(disconnect_reasons[0]);

static const wifi_disconnect_reason_t *find_disconnect_reason(wifi_err_reason_t reason) {
  for (size_t i = 0; i < disconnect_reasons_count; i++) {
    if (disconnect_reasons[i].reason == reason) {
      return &disconnect_reasons[i];
    }
  }
  return NULL;
}