#ifndef MUTEX_HELPERS_H
#define MUTEX_HELPERS_H

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "common_types.h"
#include "naila_log.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mutex timeout in milliseconds (5 seconds - allows recovery from deadlocks)
#define MUTEX_TIMEOUT_MS 5000

/**
 * @brief Execute action with mutex protection, return naila_err_t on timeout
 *
 * Usage:
 *   MUTEX_LOCK(my_mutex, TAG) {
 *       // Protected code here
 *       value = shared_state;
 *   } MUTEX_UNLOCK();
 */
#define MUTEX_LOCK(mutex, tag) \
    if (xSemaphoreTake(mutex, pdMS_TO_TICKS(MUTEX_TIMEOUT_MS))) { \
        do {

#define MUTEX_UNLOCK() \
        } while(0); \
        xSemaphoreGive(mutex); \
    } else { \
        NAILA_LOGE(tag, "Mutex timeout - potential deadlock"); \
        return NAILA_ERR_TIMEOUT; \
    }

/**
 * @brief Execute action with mutex protection, return false on timeout
 *
 * For functions returning bool.
 *
 * Usage:
 *   MUTEX_LOCK_BOOL(my_mutex, TAG) {
 *       // Protected code here
 *       result = shared_state;
 *   } MUTEX_UNLOCK_BOOL();
 */
#define MUTEX_LOCK_BOOL(mutex, tag) \
    if (xSemaphoreTake(mutex, pdMS_TO_TICKS(MUTEX_TIMEOUT_MS))) { \
        do {

#define MUTEX_UNLOCK_BOOL() \
        } while(0); \
        xSemaphoreGive(mutex); \
    } else { \
        NAILA_LOGE(tag, "Mutex timeout - potential deadlock"); \
        return false; \
    }

/**
 * @brief Execute action with mutex protection, return void on timeout
 *
 * For functions with no return value.
 *
 * Usage:
 *   MUTEX_LOCK_VOID(my_mutex, TAG) {
 *       // Protected code here
 *       shared_state = value;
 *   } MUTEX_UNLOCK_VOID();
 */
#define MUTEX_LOCK_VOID(mutex, tag) \
    if (xSemaphoreTake(mutex, pdMS_TO_TICKS(MUTEX_TIMEOUT_MS))) { \
        do {

#define MUTEX_UNLOCK_VOID() \
        } while(0); \
        xSemaphoreGive(mutex); \
    } else { \
        NAILA_LOGE(tag, "Mutex timeout - potential deadlock"); \
        return; \
    }

#ifdef __cplusplus
}
#endif

#endif // MUTEX_HELPERS_H
