#ifndef NAILA_LOG_H
#define NAILA_LOG_H

#include "common_types.h"
#include "esp_log.h"
#include "esp_app_trace.h"

#ifdef __cplusplus
extern "C" {
#endif

// Logging configuration - can be disabled in production
#ifndef NAILA_LOGGING_ENABLED
#define NAILA_LOGGING_ENABLED 1
#endif

#ifndef NAILA_USE_APP_TRACE
#define NAILA_USE_APP_TRACE 0  // Set to 1 for development with host connection
#endif

// Initialize logging system
void naila_log_init(void);

// System statistics structure
typedef struct {
  uint32_t uptime_sec;
  size_t free_heap_bytes;
  size_t min_free_heap_bytes;
  uint32_t log_messages_sent;
} system_stats_t;

// Stats monitoring task management
naila_err_t naila_stats_start_task(void);
naila_err_t naila_stats_stop_task(void);
naila_err_t naila_stats_get(system_stats_t *stats);
bool naila_stats_is_task_running(void);

#if NAILA_LOGGING_ENABLED

#if NAILA_USE_APP_TRACE
// High-performance app_trace logging for development
#define NAILA_LOGE(tag, format, ...) \
  do { \
    char buf[256]; \
    int len = snprintf(buf, sizeof(buf), "[E] %s: " format "\n", tag, ##__VA_ARGS__); \
    esp_apptrace_write(ESP_APPTRACE_DEST_JTAG, buf, len, 1000); \
  } while(0)
#define NAILA_LOGW(tag, format, ...) \
  do { \
    char buf[256]; \
    int len = snprintf(buf, sizeof(buf), "[W] %s: " format "\n", tag, ##__VA_ARGS__); \
    esp_apptrace_write(ESP_APPTRACE_DEST_JTAG, buf, len, 1000); \
  } while(0)
#define NAILA_LOGI(tag, format, ...) \
  do { \
    char buf[256]; \
    int len = snprintf(buf, sizeof(buf), "[I] %s: " format "\n", tag, ##__VA_ARGS__); \
    esp_apptrace_write(ESP_APPTRACE_DEST_JTAG, buf, len, 1000); \
  } while(0)
#define NAILA_LOGD(tag, format, ...) \
  do { \
    char buf[256]; \
    int len = snprintf(buf, sizeof(buf), "[D] %s: " format "\n", tag, ##__VA_ARGS__); \
    esp_apptrace_write(ESP_APPTRACE_DEST_JTAG, buf, len, 500); \
  } while(0)
#define NAILA_LOGV(tag, format, ...) \
  do { \
    char buf[256]; \
    int len = snprintf(buf, sizeof(buf), "[V] %s: " format "\n", tag, ##__VA_ARGS__); \
    esp_apptrace_write(ESP_APPTRACE_DEST_JTAG, buf, len, 100); \
  } while(0)
#else
// Standard ESP logging for production
#define NAILA_LOGE(tag, format, ...) ESP_LOGE(tag, format, ##__VA_ARGS__)
#define NAILA_LOGW(tag, format, ...) ESP_LOGW(tag, format, ##__VA_ARGS__)
#define NAILA_LOGI(tag, format, ...) ESP_LOGI(tag, format, ##__VA_ARGS__)
#define NAILA_LOGD(tag, format, ...) ESP_LOGD(tag, format, ##__VA_ARGS__)
#define NAILA_LOGV(tag, format, ...) ESP_LOGV(tag, format, ##__VA_ARGS__)
#endif

// Error logging with error code
#define NAILA_LOG_ERROR(tag, err, format, ...) \
  NAILA_LOGE(tag, format " (0x%x)", ##__VA_ARGS__, err)

#else
// No-op logging for maximum performance
#define NAILA_LOGE(tag, format, ...) do {} while(0)
#define NAILA_LOGW(tag, format, ...) do {} while(0)
#define NAILA_LOGI(tag, format, ...) do {} while(0)
#define NAILA_LOGD(tag, format, ...) do {} while(0)
#define NAILA_LOGV(tag, format, ...) do {} while(0)
#define NAILA_LOG_ERROR(tag, err, format, ...) do {} while(0)
#endif

// Performance timing macros (always minimal overhead)
#define NAILA_TIME_START(name) int64_t time_start_##name = esp_timer_get_time()
#define NAILA_TIME_END(tag, name) \
  NAILA_LOGD(tag, "[%s] %lld μs", #name, esp_timer_get_time() - time_start_##name)

#ifdef __cplusplus
}
#endif

#endif