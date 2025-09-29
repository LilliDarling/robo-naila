#!/bin/bash
# Erase ESP32 flash memory - WARNING: This will completely wipe the device

# Load common functions
source "$(dirname "$0")/common.sh"

print_step "Erasing ESP32 flash memory..."

# Change to firmware directory
goto_firmware_dir

# Setup ESP-IDF environment
setup_esp_idf
load_env_vars

# Auto-detect ESP32 port
PORT=$(detect_esp32_port)

# Warn user about destructive operation
print_warning "This will completely erase the ESP32 flash memory!"
print_warning "All firmware, data, and settings will be lost."
echo ""
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Erase operation cancelled."
    exit 0
fi

# Erase flash
print_info "Erasing flash on $PORT..."
print_warning "This may take a moment..."
idf.py -p "$PORT" erase-flash

print_success "Flash erased successfully!"
print_info "You will need to flash new firmware with './scripts/flash.sh'"