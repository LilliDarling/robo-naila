#include "error_handling.h"

const char *naila_err_to_string(naila_err_t err) {
  switch (err) {
  case NAILA_OK:
    return "No error";
  case NAILA_FAIL:
    return "Generic failure";
  case NAILA_ERR_INVALID_ARG:
    return "Invalid argument";
  case NAILA_ERR_NO_MEM:
    return "Out of memory";
  case NAILA_ERR_TIMEOUT:
    return "Timeout";
  case NAILA_ERR_NOT_INITIALIZED:
    return "Component not initialized";
  case NAILA_ERR_ALREADY_INITIALIZED:
    return "Component already initialized";
  case NAILA_ERR_WIFI_NOT_CONNECTED:
    return "WiFi not connected";
  case NAILA_ERR_AI_MODEL_LOAD_FAILED:
    return "AI model load failed";
  case NAILA_ERR_AUDIO_INIT_FAILED:
    return "Audio initialization failed";
  case NAILA_ERR_INFERENCE_FAILED:
    return "AI inference failed";
  default:
    return "Unknown error";
  }
}