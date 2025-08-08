#!/bin/bash
# Build script for ESP32 firmware

set -e  # Exit on any error

echo "🔧 Building ESP32 firmware..."

# Change to firmware directory (parent of scripts)
cd "$(dirname "$0")/.."

# Clean old build directory
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

# Get ESP-IDF environment, load credentials, and build in same shell context
echo "Setting up ESP-IDF environment, loading WiFi credentials, and building..."
source "/Users/j/code/esp/esp-idf/export.sh" && source ./scripts/setenv.sh && idf.py build

echo "✅ Build completed successfully!"
echo "Run './flash.sh' to flash to device"