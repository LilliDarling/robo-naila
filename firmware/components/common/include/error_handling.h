#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include "common_types.h"
#include "naila_log.h"

// Enhanced error checking macros
#define NAILA_CHECK(condition, tag, err_code, format, ...)                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      NAILA_LOG_ERROR(tag, err_code, format, ##__VA_ARGS__);                   \
      return err_code;                                                         \
    }                                                                          \
  } while (0)

#define NAILA_CHECK_NULL(ptr, tag, format, ...)                                \
  NAILA_CHECK(ptr != NULL, tag, NAILA_ERR_INVALID_ARG, format, ##__VA_ARGS__)

#define NAILA_CHECK_INIT(state, tag, component)                                \
  NAILA_CHECK(state == COMPONENT_STATE_INITIALIZED, tag,                       \
      NAILA_ERR_NOT_INITIALIZED, "Component %s not initialized", component)

// Error propagation macro with context
#define NAILA_PROPAGATE_ERROR(call, tag, context)                              \
  do {                                                                         \
    naila_err_t _err = (call);                                                 \
    if (_err != NAILA_OK) {                                                    \
      NAILA_LOG_ERROR(tag, _err, "Error in %s: %s", context, #call);           \
      return _err;                                                             \
    }                                                                          \
  } while (0)

// Error handling for ESP-IDF functions
#define NAILA_ESP_CHECK(call, tag, context)                                    \
  do {                                                                         \
    esp_err_t _esp_err = (call);                                               \
    if (_esp_err != ESP_OK) {                                                  \
      NAILA_LOG_ERROR(                                                         \
          tag, _esp_err, "ESP-IDF error in %s: %s", context, #call);           \
      return (naila_err_t)_esp_err;                                            \
    }                                                                          \
  } while (0)

// Memory allocation with error checking
#define NAILA_MALLOC_CHECK(ptr, size, tag)                                     \
  do {                                                                         \
    ptr = malloc(size);                                                        \
    NAILA_CHECK_NULL(ptr, tag, "Memory allocation failed for %d bytes", size); \
  } while (0)

// Function to convert naila_err_t to string
const char *naila_err_to_string(naila_err_t err);

#endif