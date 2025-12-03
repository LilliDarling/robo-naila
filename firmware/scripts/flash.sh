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

# Flash using esptool directly (workaround for idf.py bug)
print_info "Flashing to $PORT..."
cd build
esptool.py --chip esp32s3 -p "$PORT" -b 460800 \
    --before=default_reset --after=hard_reset \
    write_flash @flash_args
cd ..

print_success "Flash completed!"

# Start monitor
print_info "Starting monitor..."
idf.py -p "$PORT" monitor