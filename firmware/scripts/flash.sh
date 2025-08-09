#!/bin/bash
# Flash script for ESP32 firmware with auto port detection

set -e  # Exit on any error

echo "🚀 Flashing ESP32 firmware..."

# Check for --skip-build flag
SKIP_BUILD=false
if [[ "$1" == "--skip-build" ]]; then
    SKIP_BUILD=true
    echo "Skipping build step..."
fi

# Build first (unless skipped)
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo "Building firmware..."
    "$(dirname "$0")/build.sh"
else
    # Still need to check if build directory exists
    cd "$(dirname "$0")/.."
    if [ ! -d "build" ]; then
        echo "❌ No build directory found. Run './scripts/build.sh' first or remove --skip-build flag."
        exit 1
    fi
fi

# Auto-detect ESP32 port
echo "Detecting ESP32 port..."
PORT=""

# Common ESP32 port patterns on macOS
for pattern in "/dev/cu.usbmodem*" "/dev/cu.usbserial*" "/dev/cu.SLAB_USBtoUART*"; do
    for port in $pattern; do
        if [ -e "$port" ]; then
            PORT="$port"
            echo "Found ESP32 at: $PORT"
            break 2
        fi
    done
done

# If no port found, ask user
if [ -z "$PORT" ]; then
    echo "⚠️  No ESP32 device auto-detected."
    echo "Available ports:"
    ls /dev/cu.* 2>/dev/null || echo "No ports found"
    echo ""
    read -p "Enter port manually (e.g., /dev/cu.usbmodem123): " PORT
    
    if [ -z "$PORT" ] || [ ! -e "$PORT" ]; then
        echo "❌ Invalid port: $PORT"
        exit 1
    fi
fi

# Change to firmware directory
cd "$(dirname "$0")/.."

# Get ESP-IDF environment
if [ -f "/Users/j/code/esp/esp-idf/export.sh" ]; then
    source "/Users/j/code/esp/esp-idf/export.sh"
else
    echo "ERROR: ESP-IDF not found at /Users/j/code/esp/esp-idf/export.sh"
    exit 1
fi

# Load environment variables
source ./scripts/setenv.sh

# Flash and monitor
echo "Flashing to $PORT..."
idf.py -p "$PORT" flash monitor

echo "✅ Flash completed!"