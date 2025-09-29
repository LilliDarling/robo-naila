#!/bin/bash
# Monitor script for ESP32 firmware - connects to serial output

set -e  # Exit on any error

echo "📟 Starting ESP32 serial monitor..."

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

# Start monitoring
echo "Starting monitor on $PORT..."
echo "Press Ctrl+] to exit monitor"
echo "─────────────────────────────────────────"
idf.py -p "$PORT" monitor

echo "✅ Monitor session ended"