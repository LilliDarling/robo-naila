#!/bin/bash
# Monitor script for ESP32 firmware - connects to serial output

# Load common functions
source "$(dirname "$0")/common.sh"

print_step "Starting ESP32 serial monitor..."

# Change to firmware directory
goto_firmware_dir

# Setup ESP-IDF environment
setup_esp_idf
load_env_vars

# Auto-detect ESP32 port
PORT=$(detect_esp32_port)

# Start monitoring
print_info "Starting monitor on $PORT..."
print_info "Press Ctrl+] to exit monitor"
echo "─────────────────────────────────────────"
idf.py -p "$PORT" monitor

print_success "Monitor session ended"