#include "wifi.h"
#include "config.h"
#include "error_handling.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "naila_log.h"
#include <string.h>

static const char *TAG = "wifi";
static EventGroupHandle_t wifi_event_group;
static const int CONNECTED_BIT = BIT0;
static const int FAIL_BIT = BIT1;
static int retry_count = 0;
static int max_retries = 5;
static component_state_t g_state = COMPONENT_STATE_UNINITIALIZED;
static component_info_t g_info = {.name = "wifi",
    .version = "1.0.0",
    .state = COMPONENT_STATE_UNINITIALIZED};

// Task management
static TaskHandle_t wifi_task_handle = NULL;
static wifi_event_callbacks_t task_callbacks = {0};
static bool task_should_stop = false;

// Forward declarations
static void wifi_reconnection_task(void *parameters);
static naila_err_t start_reconnection_task(void);

typedef struct {
  wifi_err_reason_t reason;
  esp_log_level_t log_level;
  const char *message;
} wifi_disconnect_reason_t;

static const wifi_disconnect_reason_t disconnect_reasons[] = {
    {WIFI_REASON_AUTH_EXPIRE, ESP_LOG_ERROR,
        "Authentication expired - auth timeout"},
    {WIFI_REASON_AUTH_LEAVE, ESP_LOG_ERROR,
        "Authentication leave - deauth from AP"},
    {WIFI_REASON_ASSOC_EXPIRE, ESP_LOG_ERROR,
        "Association expired - assoc timeout"},
    {WIFI_REASON_ASSOC_TOOMANY, ESP_LOG_ERROR,
        "Too many associations - AP at capacity"},
    {WIFI_REASON_NOT_AUTHED, ESP_LOG_ERROR,
        "Not authenticated - auth required first"},
    {WIFI_REASON_NOT_ASSOCED, ESP_LOG_ERROR,
        "Not associated - assoc required first"},
    {WIFI_REASON_ASSOC_LEAVE, ESP_LOG_ERROR,
        "Association leave - disassoc from AP"},
    {WIFI_REASON_ASSOC_NOT_AUTHED, ESP_LOG_ERROR,
        "Association not authenticated"},
    {WIFI_REASON_DISASSOC_PWRCAP_BAD, ESP_LOG_ERROR,
        "Disassoc due to bad power capability"},
    {WIFI_REASON_DISASSOC_SUPCHAN_BAD, ESP_LOG_ERROR,
        "Disassoc due to bad supported channels"},
    {WIFI_REASON_BSS_TRANSITION_DISASSOC, ESP_LOG_ERROR,
        "BSS transition disassociation"},
    {WIFI_REASON_IE_INVALID, ESP_LOG_ERROR, "Invalid information element"},
    {WIFI_REASON_MIC_FAILURE, ESP_LOG_ERROR, "MIC failure - encryption issue"},
    {WIFI_REASON_4WAY_HANDSHAKE_TIMEOUT, ESP_LOG_ERROR,
        "4-way handshake timeout - WPA issue"},
    {WIFI_REASON_GROUP_KEY_UPDATE_TIMEOUT, ESP_LOG_ERROR,
        "Group key update timeout"},
    {WIFI_REASON_IE_IN_4WAY_DIFFERS, ESP_LOG_ERROR,
        "IE in 4-way handshake differs"},
    {WIFI_REASON_GROUP_CIPHER_INVALID, ESP_LOG_ERROR, "Invalid group cipher"},
    {WIFI_REASON_PAIRWISE_CIPHER_INVALID, ESP_LOG_ERROR,
        "Invalid pairwise cipher"},
    {WIFI_REASON_AKMP_INVALID, ESP_LOG_ERROR, "Invalid AKMP (auth/key mgmt)"},
    {WIFI_REASON_UNSUPP_RSN_IE_VERSION, ESP_LOG_ERROR,
        "Unsupported RSN IE version"},
    {WIFI_REASON_INVALID_RSN_IE_CAP, ESP_LOG_ERROR,
        "Invalid RSN IE capabilities"},
    {WIFI_REASON_802_1X_AUTH_FAILED, ESP_LOG_ERROR,
        "802.1X authentication failed"},
    {WIFI_REASON_CIPHER_SUITE_REJECTED, ESP_LOG_ERROR, "Cipher suite rejected"},
    {WIFI_REASON_BEACON_TIMEOUT, ESP_LOG_ERROR,
        "Beacon timeout - lost connection to AP"},
    {WIFI_REASON_NO_AP_FOUND, ESP_LOG_ERROR, "No AP found - SSID not visible"},
    {WIFI_REASON_AUTH_FAIL, ESP_LOG_ERROR,
        "Authentication failed - wrong credentials"},
    {WIFI_REASON_ASSOC_FAIL, ESP_LOG_ERROR,
        "Association failed - AP rejected association"},
    {WIFI_REASON_HANDSHAKE_TIMEOUT, ESP_LOG_ERROR,
        "Handshake timeout - general timeout"},
    {WIFI_REASON_CONNECTION_FAIL, ESP_LOG_ERROR,
        "Connection failed - general failure"},
    {WIFI_REASON_AP_TSF_RESET, ESP_LOG_ERROR, "AP TSF reset"},
    {WIFI_REASON_ROAMING, ESP_LOG_INFO, "Roaming to another AP"}};

static const size_t disconnect_reasons_count =
    sizeof(disconnect_reasons) / sizeof(disconnect_reasons[0]);

static const wifi_disconnect_reason_t *find_disconnect_reason(
    wifi_err_reason_t reason) {
  for (size_t i = 0; i < disconnect_reasons_count; i++) {
    if (disconnect_reasons[i].reason == reason) {
      return &disconnect_reasons[i];
    }
  }
  return NULL;
}

static void event_handler(void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data) {

  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
  } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
    ESP_LOGI(TAG, "WiFi connected successfully!");
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_AUTHMODE_CHANGE) {
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_event_sta_disconnected_t *disconnected_event =
        (wifi_event_sta_disconnected_t *)event_data;

    // Log the specific disconnect reason with detailed explanations
    const wifi_disconnect_reason_t *reason_info =
        find_disconnect_reason(disconnected_event->reason);
    if (reason_info != NULL) {
      ESP_LOG_LEVEL(reason_info->log_level, TAG, "%s", reason_info->message);
    } else {
      ESP_LOGE(
          TAG, "Unknown disconnect reason: %d", disconnected_event->reason);
    }

    // Clear connected bit and start reconnection task
    xEventGroupClearBits(wifi_event_group, CONNECTED_BIT);

    // Start reconnection task (only if not already running)
    ESP_LOGI(TAG, "Starting reconnection task due to disconnect...");
    naila_err_t result = start_reconnection_task();
    if (result != NAILA_OK) {
      ESP_LOGE(TAG, "Failed to start reconnection task: %d", result);
      xEventGroupSetBits(wifi_event_group, FAIL_BIT);
    } else {
      ESP_LOGI(TAG, "Reconnection task started successfully");
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    retry_count = 0;
    xEventGroupSetBits(wifi_event_group, CONNECTED_BIT);

    // Stop reconnection task since we're now connected
    if (wifi_task_handle != NULL) {
      ESP_LOGI(TAG, "🛑 Stopping reconnection task due to successful connection");
      task_should_stop = true;
    }

    // Call the connected callback
    if (task_callbacks.on_connected) {
      task_callbacks.on_connected();
    }
  }
}

naila_err_t wifi_init(void) {
  NAILA_LOGI(TAG, "WiFi manager init starting...");

  if (g_state == COMPONENT_STATE_INITIALIZED) {
    return NAILA_ERR_ALREADY_INITIALIZED;
  }

  g_state = COMPONENT_STATE_INITIALIZING;
  g_info.state = COMPONENT_STATE_INITIALIZING;

  wifi_reset_connection_state();

  wifi_event_group = xEventGroupCreate();
  NAILA_CHECK_NULL(wifi_event_group, TAG, "Failed to create WiFi event group");

  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  NAILA_ESP_CHECK(esp_wifi_init(&cfg), TAG, "WiFi init");

  NAILA_ESP_CHECK(
      esp_wifi_set_country_code("US", true), TAG, "Set country code");

  NAILA_LOGI(TAG, "Registering event handlers...");
  NAILA_ESP_CHECK(esp_event_handler_instance_register(
                      WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL),
      TAG, "Register WiFi event handler");
  NAILA_ESP_CHECK(esp_event_handler_instance_register(IP_EVENT,
                      IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL),
      TAG, "Register IP event handler");
  NAILA_LOGI(TAG, "Event handlers registered successfully");

  g_state = COMPONENT_STATE_INITIALIZED;
  g_info.state = COMPONENT_STATE_INITIALIZED;
  return NAILA_OK;
}

naila_err_t wifi_connect(const wifi_config_simple_t *config) {
  NAILA_CHECK_NULL(config, TAG, "Config pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "wifi");

  // Skip pre-scan - let ESP32 internal stack handle AP discovery during
  // connection

  // Minimal config - let ESP32 use defaults for hotspot compatibility
  wifi_config_t wifi_config = {
      .sta =
          {
              .threshold.authmode = WIFI_AUTH_WPA2_PSK,
              .pmf_cfg = {.capable = true, .required = false},
          },
  };

  // Set the SSID and password from the config
  strlcpy(
      (char *)wifi_config.sta.ssid, config->ssid, sizeof(wifi_config.sta.ssid));
  strlcpy((char *)wifi_config.sta.password, config->password,
      sizeof(wifi_config.sta.password));

  NAILA_LOGI(TAG, "Connecting to SSID: %s", wifi_config.sta.ssid);

  max_retries = config->max_retry;
  retry_count = 0;

  // Only disable power saving - let ESP32 use optimal defaults for everything
  // else
  esp_wifi_set_ps(WIFI_PS_NONE);

  NAILA_ESP_CHECK(esp_wifi_set_mode(WIFI_MODE_STA), TAG, "Set WiFi mode");
  NAILA_ESP_CHECK(
      esp_wifi_set_config(WIFI_IF_STA, &wifi_config), TAG, "Set WiFi config");
  NAILA_ESP_CHECK(esp_wifi_start(), TAG, "Start WiFi interface");

  // Brief delay for WiFi to initialize
  vTaskDelay(pdMS_TO_TICKS(1000));

  esp_wifi_connect();

  // Wait for connection result
  EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
      CONNECTED_BIT | FAIL_BIT, pdFALSE, pdFALSE, portMAX_DELAY);

  return (bits & CONNECTED_BIT) ? NAILA_OK : NAILA_ERR_WIFI_NOT_CONNECTED;
}

bool wifi_is_connected(void) {
  EventBits_t bits = xEventGroupGetBits(wifi_event_group);
  return (bits & CONNECTED_BIT) != 0;
}

naila_err_t wifi_disconnect(void) {
  NAILA_CHECK_INIT(g_state, TAG, "wifi");

  esp_err_t disconnect_result = esp_wifi_disconnect();
  if (disconnect_result == ESP_OK) {
    vTaskDelay(pdMS_TO_TICKS(500));
  }

  NAILA_ESP_CHECK(esp_wifi_stop(), TAG, "Stop WiFi");

  // Clear event group bits
  if (wifi_event_group != NULL) {
    xEventGroupClearBits(wifi_event_group, CONNECTED_BIT | FAIL_BIT);
  }

  // Reset retry counter
  retry_count = 0;

  return NAILA_OK;
}

// ADDED: Complete WiFi deinit function for proper cleanup - can be removed if
// causing issues
naila_err_t wifi_cleanup(void) {

  if (g_state == COMPONENT_STATE_UNINITIALIZED) {
    return NAILA_OK;
  }

  if (wifi_is_connected()) {
    wifi_disconnect();
  }

  // Unregister event handlers
  esp_event_handler_instance_unregister(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler);
  esp_event_handler_instance_unregister(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler);

  // Deinitialize WiFi
  esp_wifi_deinit();

  // Clean up event group
  if (wifi_event_group != NULL) {
    vEventGroupDelete(wifi_event_group);
    wifi_event_group = NULL;
  }

  // Reset state
  g_state = COMPONENT_STATE_UNINITIALIZED;
  g_info.state = COMPONENT_STATE_UNINITIALIZED;
  retry_count = 0;

  return NAILA_OK;
}
// END ADDED

// ADDED: WiFi connection state reset function for reliable reconnection - can
// be removed if causing issues
naila_err_t wifi_reset_connection_state(void) {

  // Reset retry counter
  retry_count = 0;

  // Clear event group bits if group exists
  if (wifi_event_group != NULL) {
    xEventGroupClearBits(wifi_event_group, CONNECTED_BIT | FAIL_BIT);
  }

  esp_wifi_stop();

  // Brief delay to ensure WiFi stack resets
  vTaskDelay(pdMS_TO_TICKS(1000));

  return NAILA_OK;
}
// END ADDED

naila_err_t wifi_get_info(component_info_t *info) {
  NAILA_CHECK_NULL(info, TAG, "Info pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "wifi");

  memcpy(info, &g_info, sizeof(component_info_t));
  info->state = g_state;
  return NAILA_OK;
}

// Helper function to start reconnection task if not already running
static naila_err_t start_reconnection_task(void) {
  ESP_LOGI(TAG, "🔍 start_reconnection_task: wifi_task_handle = %p", wifi_task_handle);

  if (wifi_task_handle != NULL) {
    ESP_LOGI(TAG, "⚠️ Reconnection task already running, skipping");
    return NAILA_OK; // Task already running
  }

  ESP_LOGI(TAG, "✨ Creating new reconnection task");
  task_should_stop = false;
  BaseType_t result = xTaskCreate(
      wifi_reconnection_task, "wifi_reconnect", 4096, NULL, 5, &wifi_task_handle);

  if (result == pdPASS) {
    ESP_LOGI(TAG, "✅ Reconnection task created, handle = %p", wifi_task_handle);
  } else {
    ESP_LOGE(TAG, "❌ Failed to create reconnection task");
  }

  return (result == pdPASS) ? NAILA_OK : NAILA_ERR_NO_MEM;
}

// Reconnection task - only runs when trying to connect
static void wifi_reconnection_task(void *parameters) {
  ESP_LOGI(TAG, "🔄 Reconnection task started");
  const naila_config_t *config = config_manager_get();
  if (config == NULL) {
    if (task_callbacks.on_error) {
      task_callbacks.on_error(NAILA_ERR_INVALID_ARG);
    }
    wifi_task_handle = NULL;
    vTaskDelete(NULL);
    return;
  }

  wifi_config_simple_t wifi_config = {.ssid = config->wifi.ssid,
      .password = config->wifi.password,
      .max_retry = config->wifi.max_retry};

  int attempt = 0;
  while (!task_should_stop && !wifi_is_connected() && attempt < wifi_config.max_retry) {
    attempt++; // Increment attempt at start of each iteration
    ESP_LOGI(TAG, "🔄 Connection attempt %d/%d", attempt, wifi_config.max_retry);

    naila_err_t connect_result = wifi_connect(&wifi_config);
    if (connect_result != NAILA_OK) {
      if (attempt >= wifi_config.max_retry) {
        if (task_callbacks.on_error) {
          task_callbacks.on_error(NAILA_ERR_WIFI_NOT_CONNECTED);
        }
        break;
      }
      vTaskDelay(pdMS_TO_TICKS(5000)); // Wait before retry
    } else {
      // Connection attempt started, wait for result
      vTaskDelay(pdMS_TO_TICKS(10000)); // Wait for connection to establish

      // If still not connected after waiting, this attempt failed
      if (!wifi_is_connected() && attempt >= wifi_config.max_retry) {
        if (task_callbacks.on_error) {
          task_callbacks.on_error(NAILA_ERR_WIFI_NOT_CONNECTED);
        }
        break;
      }
    }
  }

  ESP_LOGI(TAG, "🏁 Reconnection task exiting, setting handle to NULL");
  wifi_task_handle = NULL;
  vTaskDelete(NULL);
}

naila_err_t wifi_start_task(const wifi_event_callbacks_t *callbacks) {
  NAILA_CHECK_INIT(g_state, TAG, "wifi");

  if (callbacks) {
    task_callbacks = *callbacks;
  }

  // Start reconnection task if not connected (for initial connection)
  if (!wifi_is_connected()) {
    return start_reconnection_task();
  }

  return NAILA_OK;
}

naila_err_t wifi_stop_task(void) {
  if (wifi_task_handle == NULL) {
    return NAILA_OK;
  }

  task_should_stop = true;
  vTaskDelay(pdMS_TO_TICKS(1000)); // Wait for task to exit

  if (wifi_task_handle != NULL) {
    vTaskDelete(wifi_task_handle);
    wifi_task_handle = NULL;
  }

  return NAILA_OK;
}

bool wifi_is_task_running(void) { return wifi_task_handle != NULL; }