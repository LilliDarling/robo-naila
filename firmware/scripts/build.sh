#!/bin/bash
# Build script for ESP32 firmware

# Load common functions
source "$(dirname "$0")/common.sh"

print_step "Building ESP32 firmware..."

# Change to firmware directory
goto_firmware_dir

# Clean old build directory
if [ -d "build" ]; then
    print_info "Removing old build directory..."
    rm -rf build
fi

# Setup environment and build
print_info "Setting up ESP-IDF environment and loading credentials..."
setup_esp_idf
load_env_vars

print_info "Starting build..."
idf.py build

print_success "Build completed successfully!"
echo "Run './scripts/flash.sh' to flash to device"