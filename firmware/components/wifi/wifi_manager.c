#include "wifi_manager.h"
#include "error_handling.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "naila_log.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include <string.h>

static const char *TAG = "wifi_manager";
static EventGroupHandle_t wifi_event_group;
static const int CONNECTED_BIT = BIT0;
static const int FAIL_BIT = BIT1;
static int retry_count = 0;
static int max_retries = 5;
static component_state_t g_state = COMPONENT_STATE_UNINITIALIZED;
static component_info_t g_info = {.name = "wifi_manager",
    .version = "1.0.0",
    .state = COMPONENT_STATE_UNINITIALIZED};

typedef struct {
  wifi_err_reason_t reason;
  esp_log_level_t log_level;
  const char* message;
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

static const wifi_disconnect_reason_t* find_disconnect_reason(wifi_err_reason_t reason) {
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
    ESP_LOGI(TAG, "WiFi started - waiting for manual connection command...");

    // Get and log WiFi MAC address
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Station MAC: %02x:%02x:%02x:%02x:%02x:%02x", mac[0], mac[1],
        mac[2], mac[3], mac[4], mac[5]);

    // Don't auto-connect - let our scan logic handle it
  } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
    wifi_event_sta_connected_t *connected =
        (wifi_event_sta_connected_t *)event_data;
    ESP_LOGI(TAG, "Connected to AP! SSID: %s, Channel: %d, Auth: %d",
        connected->ssid, connected->channel, connected->authmode);
  } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_AUTHMODE_CHANGE) {
    wifi_event_sta_authmode_change_t *authmode_change = (wifi_event_sta_authmode_change_t *)event_data;
    ESP_LOGI(TAG, "WiFi authmode changed: old=%d, new=%d", authmode_change->old_mode, authmode_change->new_mode);
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_event_sta_disconnected_t *disconnected_event =
        (wifi_event_sta_disconnected_t *)event_data;
    ESP_LOGI(TAG, "WiFi disconnected, reason: %d", disconnected_event->reason);

    // Log the specific disconnect reason with detailed explanations
    const wifi_disconnect_reason_t* reason_info = find_disconnect_reason(disconnected_event->reason);
    if (reason_info != NULL) {
      ESP_LOG_LEVEL(reason_info->log_level, TAG, "%s", reason_info->message);
    } else {
      ESP_LOGE(TAG, "Unknown disconnect reason: %d", disconnected_event->reason);
    }

    if (retry_count < max_retries) {
      ESP_LOGI(
          TAG, "Retrying WiFI connection %d/%d", retry_count + 1, max_retries);
      retry_count++;
      
      // Add progressive delay for hotspot association issues
      int delay_ms = 2000 + (retry_count * 1000);  // 2s, 3s, 4s, 5s, 6s
      ESP_LOGI(TAG, "Waiting %d ms before retry for association timeout", delay_ms);
      vTaskDelay(pdMS_TO_TICKS(delay_ms));
      
      esp_wifi_connect();
    } else {
      ESP_LOGE(TAG, "WiFi connection failed after %d retries", max_retries);
      xEventGroupSetBits(wifi_event_group, FAIL_BIT);
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    ESP_LOGI(TAG, "Successfully connected! Got IP address: " IPSTR,
        IP2STR(&event->ip_info.ip));
    retry_count = 0;
    xEventGroupSetBits(wifi_event_group, CONNECTED_BIT);
  }
}

naila_err_t wifi_manager_init(void) {
  NAILA_LOG_FUNC_ENTER(TAG);

  // Enable maximum WiFi debugging
  esp_log_level_set("wifi", ESP_LOG_VERBOSE);
  esp_log_level_set("wifi_init", ESP_LOG_VERBOSE);
  esp_log_level_set("phy_init", ESP_LOG_VERBOSE);
  esp_log_level_set("wifi_station", ESP_LOG_VERBOSE);
  esp_log_level_set("wifi_conn", ESP_LOG_VERBOSE);
  esp_log_level_set("wpa", ESP_LOG_VERBOSE);
  esp_log_level_set("wpa_supplicant", ESP_LOG_VERBOSE);
  esp_log_level_set("net80211", ESP_LOG_VERBOSE);
  esp_log_level_set("ieee80211", ESP_LOG_VERBOSE);
  esp_log_level_set("wpa2", ESP_LOG_VERBOSE);

  if (g_state == COMPONENT_STATE_INITIALIZED) {
    return NAILA_ERR_ALREADY_INITIALIZED;
  }

  g_state = COMPONENT_STATE_INITIALIZING;
  g_info.state = COMPONENT_STATE_INITIALIZING;

  // ADDED: Reset connection state completely to handle previous unclean shutdowns - can be removed if causing issues
  NAILA_LOGI(TAG, "Resetting WiFi state for clean initialization");
  wifi_manager_reset_connection_state();
  // END ADDED

  wifi_event_group = xEventGroupCreate();
  NAILA_CHECK_NULL(wifi_event_group, TAG, "Failed to create WiFi event group");

  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  NAILA_ESP_CHECK(esp_wifi_init(&cfg), TAG, "WiFi init");

  // Set country code for proper RF regulatory compliance
  // This is crucial for ESP32-S3 association issues
  NAILA_LOGI(TAG, "Setting WiFi country code to US");
  NAILA_ESP_CHECK(
      esp_wifi_set_country_code("US", true), TAG, "Set country code");

  NAILA_ESP_CHECK(esp_event_handler_instance_register(
                      WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL),
      TAG, "Register WiFi event handler");
  NAILA_ESP_CHECK(esp_event_handler_instance_register(IP_EVENT,
                      IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL),
      TAG, "Register IP event handler");

  g_state = COMPONENT_STATE_INITIALIZED;
  g_info.state = COMPONENT_STATE_INITIALIZED;
  NAILA_LOGI(TAG, "WiFi manager initialized successfully");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

naila_err_t wifi_manager_connect(const wifi_config_simple_t *config) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_NULL(config, TAG, "Config pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "wifi_manager");

  // Skip pre-scan - let ESP32 internal stack handle AP discovery during connection

  // Minimal config - let ESP32 use defaults for hotspot compatibility
  wifi_config_t wifi_config = {
      .sta = {
          .threshold.authmode = WIFI_AUTH_OPEN,
          .pmf_cfg = {
              .capable = false,
              .required = false
          },
      },
  };

  // Set the SSID and password from the config
  strlcpy((char *)wifi_config.sta.ssid, config->ssid, sizeof(wifi_config.sta.ssid));
  strlcpy((char *)wifi_config.sta.password, config->password, sizeof(wifi_config.sta.password));

  NAILA_LOGI(TAG, "WiFi config - SSID: %s, Password: '%s' (length: %d)",
      wifi_config.sta.ssid, wifi_config.sta.password, strlen((char *)wifi_config.sta.password));

  max_retries = config->max_retry;
  retry_count = 0;

  NAILA_LOGI(TAG, "Using minimal WiFi settings for hotspot compatibility");
  
  // Only disable power saving - let ESP32 use optimal defaults for everything else
  esp_wifi_set_ps(WIFI_PS_NONE);
  
  NAILA_ESP_CHECK(esp_wifi_set_mode(WIFI_MODE_STA), TAG, "Set WiFi mode");
  NAILA_ESP_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config), TAG, "Set WiFi config");
  NAILA_ESP_CHECK(esp_wifi_start(), TAG, "Start WiFi interface");
  
  // Brief delay for WiFi to initialize
  vTaskDelay(pdMS_TO_TICKS(1000));
  
  NAILA_LOGI(TAG, "Attempting connection with default settings...");
  esp_wifi_connect();

  NAILA_LOGI(TAG, "Connection initiated, waiting for result...");

  // Wait for connection result
  EventBits_t bits = xEventGroupWaitBits(wifi_event_group,
      CONNECTED_BIT | FAIL_BIT, pdFALSE, pdFALSE, portMAX_DELAY);

  naila_err_t result = (bits & CONNECTED_BIT) ? NAILA_OK : NAILA_ERR_WIFI_NOT_CONNECTED;

  NAILA_LOG_FUNC_EXIT(TAG);
  return result;
}

bool wifi_manager_is_connected(void) {
  EventBits_t bits = xEventGroupGetBits(wifi_event_group);
  return (bits & CONNECTED_BIT) != 0;
}

naila_err_t wifi_manager_disconnect(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  NAILA_CHECK_INIT(g_state, TAG, "wifi_manager");

  // ADDED: Enhanced graceful disconnection to fix hotspot connection persistence - can be removed if causing issues
  // First disconnect gracefully to send deauth frames
  esp_err_t disconnect_result = esp_wifi_disconnect();
  if (disconnect_result == ESP_OK) {
    NAILA_LOGI(TAG, "Graceful WiFi disconnect initiated");
    // Wait briefly for deauth frames to be sent
    vTaskDelay(pdMS_TO_TICKS(500));
  } else {
    NAILA_LOGW(TAG, "Graceful disconnect failed: %s, forcing stop", esp_err_to_name(disconnect_result));
  }

  NAILA_ESP_CHECK(esp_wifi_stop(), TAG, "Stop WiFi");
  
  // Clear event group bits
  if (wifi_event_group != NULL) {
    xEventGroupClearBits(wifi_event_group, CONNECTED_BIT | FAIL_BIT);
  }
  
  // Reset retry counter
  retry_count = 0;
  
  NAILA_LOGI(TAG, "WiFi disconnected and state cleared");
  // END ADDED
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}

// ADDED: Complete WiFi deinit function for proper cleanup - can be removed if causing issues
naila_err_t wifi_manager_deinit(void) {
  NAILA_LOG_FUNC_ENTER(TAG);
  
  if (g_state == COMPONENT_STATE_UNINITIALIZED) {
    return NAILA_OK;
  }

  // Gracefully disconnect if connected
  if (wifi_manager_is_connected()) {
    NAILA_LOGI(TAG, "Disconnecting WiFi before deinit");
    wifi_manager_disconnect();
  }

  // Unregister event handlers
  esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler);
  esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler);

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

  NAILA_LOGI(TAG, "WiFi manager deinitialized");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}
// END ADDED

// ADDED: WiFi connection state reset function for reliable reconnection - can be removed if causing issues
naila_err_t wifi_manager_reset_connection_state(void) {
  NAILA_LOG_FUNC_ENTER(TAG);

  // Reset retry counter
  retry_count = 0;
  
  // Clear event group bits if group exists
  if (wifi_event_group != NULL) {
    xEventGroupClearBits(wifi_event_group, CONNECTED_BIT | FAIL_BIT);
  }

  // Force WiFi to stop and reset internal state
  esp_err_t stop_result = esp_wifi_stop();
  if (stop_result != ESP_OK && stop_result != ESP_ERR_WIFI_NOT_INIT) {
    NAILA_LOGW(TAG, "WiFi stop during reset failed: %s", esp_err_to_name(stop_result));
  }

  // Brief delay to ensure WiFi stack resets
  vTaskDelay(pdMS_TO_TICKS(1000));

  NAILA_LOGI(TAG, "WiFi connection state reset completed");
  NAILA_LOG_FUNC_EXIT(TAG);
  return NAILA_OK;
}
// END ADDED

naila_err_t wifi_manager_get_info(component_info_t *info) {
  NAILA_CHECK_NULL(info, TAG, "Info pointer is null");
  NAILA_CHECK_INIT(g_state, TAG, "wifi_manager");

  memcpy(info, &g_info, sizeof(component_info_t));
  info->state = g_state;
  return NAILA_OK;
}