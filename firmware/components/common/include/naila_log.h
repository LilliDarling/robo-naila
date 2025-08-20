#ifndef NAILA_LOG_H
#define NAILA_LOG_H

#include "common_types.h"
#include "esp_log.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enhanced logging macros with component context
#define NAILA_LOGE(tag, format, ...)                                           \
  ESP_LOGE(tag, "[ERROR] " format, ##__VA_ARGS__)
#define NAILA_LOGW(tag, format, ...)                                           \
  ESP_LOGW(tag, "[WARN]  " format, ##__VA_ARGS__)
#define NAILA_LOGI(tag, format, ...)                                           \
  ESP_LOGI(tag, "[INFO]  " format, ##__VA_ARGS__)
#define NAILA_LOGD(tag, format, ...)                                           \
  ESP_LOGD(tag, "[DEBUG] " format, ##__VA_ARGS__)
#define NAILA_LOGV(tag, format, ...)                                           \
  ESP_LOGV(tag, "[VERBOSE] " format, ##__VA_ARGS__)

// Error logging with error code
#define NAILA_LOG_ERROR(tag, err, format, ...)                                 \
  NAILA_LOGE(tag, format " (error: 0x%x)", ##__VA_ARGS__, err)

// Function entry/exit logging for debugging
#define NAILA_LOG_FUNC_ENTER(tag) NAILA_LOGV(tag, "-> %s", __func__)
#define NAILA_LOG_FUNC_EXIT(tag) NAILA_LOGV(tag, "<- %s", __func__)

// Component state logging
#define NAILA_LOG_COMPONENT_STATE(tag, component, state)                       \
  NAILA_LOGI(tag, "Component %s state: %d", component, state)

// Performance timing macros
#define NAILA_TIME_START(name) int64_t time_start_##name = esp_timer_get_time()

#define NAILA_TIME_END(tag, name)                                              \
  NAILA_LOGD(tag, "Timing [%s]: %lld μs", #name,                               \
      esp_timer_get_time() - time_start_##name)

#ifdef __cplusplus
}
#endif

#endif