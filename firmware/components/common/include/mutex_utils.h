#ifndef MUTEX_UTILS_H
#define MUTEX_UTILS_H

#include "common_types.h"
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#ifdef __cplusplus
extern "C" {
#endif

// Default mutex timeout (100ms)
static const TickType_t MUTEX_DEFAULT_TIMEOUT = pdMS_TO_TICKS(100);

// Callback function type for mutex-protected operations
// Returns NAILA_OK on success, error code on failure
typedef naila_err_t (*mutex_callback_t)(void* context);

// Execute a callback function with mutex protection
// Parameters:
//   mutex: The semaphore/mutex to acquire
//   callback: Function to execute while holding the mutex
//   context: User data passed to the callback
//   timeout: Maximum time to wait for mutex acquisition
// Returns:
//   NAILA_OK if callback executed successfully
//   NAILA_ERR_TIMEOUT if mutex could not be acquired
//   Error code from callback if it fails
naila_err_t mutex_execute(SemaphoreHandle_t mutex,
                          mutex_callback_t callback,
                          void* context);

#ifdef __cplusplus
}
#endif

#endif // MUTEX_UTILS_H
