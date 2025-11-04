#!/bin/bash
# Common functions for ESP32 firmware scripts

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Auto-detect ESP32 port
detect_esp32_port() {
    local PORT=""

    print_info "Detecting ESP32 port..."

    # Common ESP32 port patterns on macOS
    for pattern in "/dev/cu.usbmodem*" "/dev/cu.usbserial*" "/dev/cu.SLAB_USBtoUART*"; do
        for port in $pattern; do
            if [ -e "$port" ]; then
                PORT="$port"
                print_success "Found ESP32 at: $PORT"
                echo "$PORT"
                return 0
            fi
        done
    done

    # If no port found, ask user
    print_warning "No ESP32 device auto-detected."
    echo "Available ports:"
    ls /dev/cu.* 2>/dev/null || echo "No ports found"
    echo ""
    read -p "Enter port manually (e.g., /dev/cu.usbmodem123): " PORT

    if [ -z "$PORT" ] || [ ! -e "$PORT" ]; then
        print_error "Invalid port: $PORT"
        exit 1
    fi

    echo "$PORT"
}

# Setup ESP-IDF environment
setup_esp_idf() {
    # Try common ESP-IDF installation paths
    local ESP_IDF_PATHS=(
        "/Users/j/code/esp/esp-idf/export.sh"
        "$HOME/esp/esp-idf/export.sh"
        "$IDF_PATH/export.sh"
    )

    for path in "${ESP_IDF_PATHS[@]}"; do
        if [ -f "$path" ]; then
            print_info "Loading ESP-IDF from: $path"
            source "$path"
            return 0
        fi
    done

    print_error "ESP-IDF not found. Please set IDF_PATH or install ESP-IDF."
    exit 1
}

# Change to firmware directory
goto_firmware_dir() {
    cd "$(dirname "$0")/.."
}

# Load environment variables
load_env_vars() {
    if [ -f "./scripts/setenv.sh" ]; then
        source ./scripts/setenv.sh
    else
        print_warning "setenv.sh not found, skipping environment variable loading"
    fi
}

# Check if build directory exists
check_build_exists() {
    if [ ! -d "build" ]; then
        print_error "No build directory found. Run './scripts/build.sh' first."
        exit 1
    fi
}