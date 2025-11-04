#!/bin/bash
# Flash script for ESP32 firmware with auto port detection

# Load common functions
source "$(dirname "$0")/common.sh"

print_step "Flashing ESP32 firmware..."

# Check for --skip-build flag
SKIP_BUILD=false
if [[ "$1" == "--skip-build" ]]; then
    SKIP_BUILD=true
    print_info "Skipping build step..."
fi

# Change to firmware directory
goto_firmware_dir

# Build first (unless skipped)
if [[ "$SKIP_BUILD" == "false" ]]; then
    print_info "Building firmware..."
    "$(dirname "$0")/build.sh"
else
    # Still need to check if build directory exists
    check_build_exists
fi

# Setup ESP-IDF environment
setup_esp_idf
load_env_vars

# Auto-detect ESP32 port
PORT=$(detect_esp32_port)

# Flash and monitor
print_info "Flashing to $PORT..."
idf.py -p "$PORT" flash monitor

print_success "Flash completed!"