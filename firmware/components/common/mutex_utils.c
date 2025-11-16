#include "mutex_utils.h"

naila_err_t mutex_execute(SemaphoreHandle_t mutex,
                          mutex_callback_t callback,
                          void* context) {
    if (!mutex || !callback) {
        return NAILA_ERR_INVALID_ARG;
    }

    if (xSemaphoreTake(mutex, MUTEX_DEFAULT_TIMEOUT) != pdTRUE) {
        return NAILA_ERR_TIMEOUT;
    }

    naila_err_t result = callback(context);

    xSemaphoreGive(mutex);

    return result;
}
