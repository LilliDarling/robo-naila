# ESP32 Firmware Scripts

This directory contains helper scripts for ESP32 firmware development.

## Available Scripts

- **`build.sh`** - Build the ESP32 firmware
- **`flash.sh`** - Flash firmware to device (includes auto port detection)
  - Use `--skip-build` flag to skip building if firmware is already built
- **`monitor.sh`** - Monitor serial output from the device
- **`erase.sh`** - Erase ESP32 flash memory (destructive operation)
- **`format.sh`** - Format all C/C++ files using clang-format
- **`setenv.sh`** - Load environment variables from .env file
- **`common.sh`** - Shared functions used by other scripts

## Usage

All scripts should be run from anywhere in the project:

```bash
# Build firmware
./scripts/build.sh

# Flash to device (builds first, then flashes and monitors)
./scripts/flash.sh

# Flash without building (if already built)
./scripts/flash.sh --skip-build

# Just monitor device
./scripts/monitor.sh

# Erase device flash (WARNING: destructive)
./scripts/erase.sh

# Format code
./scripts/format.sh
```

## Port Auto-Detection

The flash, monitor, and erase scripts automatically detect the ESP32 USB port. If multiple devices are connected or auto-detection fails, you'll be prompted to enter the port manually.

## Requirements

- ESP-IDF properly installed and configured
- `.env` file in project root with WiFi and MQTT credentials
- clang-format installed (for formatting script)