| Supported Target | ESP32-S3 |
| ---------------- | -------- |

# NAILA Robot Firmware

This directory contains the firmware source code for the robot's embedded system, designed to run on the Seeed Studio XIAO ESP32-S3 Sense board. It is built using the Espressif IDF (IoT Development Framework) and manages the robot's hardware interactions, on-device processing, and communication with the distributed AI orchestration system.

## ⚠️ Important Note: Environment Setup

Before you can build this firmware, you *must* have the ESP-IDF development environment properly set up on your system (Linux, macOS, or Windows). This includes:

* Installing necessary system-level dependencies (e.g., `git`, `cmake`, `ninja`, `dfu-util`, `libusb` and development files).
* Cloning the ESP-IDF repository.
* Running the `install.sh` (or equivalent) script to download and configure the Xtensa toolchain and Python virtual environment.
* Sourcing the `export.sh` script to set up your environment variables.

Refer to the official ESP-IDF "Get Started" guide for detailed instructions specific to your operating system:
[Espressif Docs](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/index.html)  


## Project Structure

This is a standard ESP-IDF project structure aligned with the new multi-agent architecture. Key directories:

* **`main/`**: Contains the core application logic of the robot.
    * `robot_main.c`: The main entry point and overall control flow.
    * `communications/`: Handles Wi-Fi connectivity, MQTT client implementation for communicating with the AI orchestration layer, and OTA update client logic.
    * `peripherals/`: Drivers and abstraction layers for the board's hardware:
        * `audio_processing.c`: Real-time audio capture and preprocessing
        * `sensor_manager.c`: Unified interface for touch, IR, and environmental sensors
        * `vision_module.c`: On-device computer vision processing and camera management
        * `motor_control.c`: Servo coordination, movement patterns, and gesture execution
        * `display_driver.c`: Screen control for expressive animations
    * `ai_inference/`: Contains on-device AI capabilities:
        * Wake word detection models and inference
        * Basic vision preprocessing
        * Local safety and emergency response logic
    * `security/`: Device-level security implementations
    * ... (other robot-specific logic as it evolves)
* **`components/`**: Custom components or Git submodules for external C/C++ libraries not part of the ESP-IDF components registry.
* **`idf_component.yml`**: Defines dependencies on components from the Espressif Components Registry. The ESP-IDF build system automatically downloads and manages these.
* **`partitions.csv`**: Defines the flash memory layout of the ESP32-S3. **Crucial for Over-The-Air (OTA) updates**, as it defines multiple application partitions.
* **`CMakeLists.txt`**: The build system configuration file for this project. It defines how source files are compiled and linked.
* **`sdkconfig.defaults`**: Default configuration options for `menuconfig`. Useful for maintaining consistent settings.

## Configuration Setup

Before building, you need to configure WiFi credentials:

1. **Create environment file:**
   Create a `.env` file in the project root (`../env`) with your WiFi credentials:
   ```bash
   # WiFi Configuration
   WIFI_SSID="your_network_name"
   WIFI_PASSWORD="your_password"
   ```
   
   ⚠️ **Important**: The `.env` file is gitignored and should never be committed to version control.

## Building and Flashing

This project includes automated build scripts that handle ESP-IDF environment setup and WiFi configuration.

### Quick Start (Automated)

1. **Navigate to the firmware directory:**
   ```bash
   cd firmware/
   ```

2. **Connect your ESP32-S3 board** via USB cable.

3. **Build and flash (recommended):**
   ```bash
   ./scripts/flash.sh
   ```
   This script will:
   - Set up the ESP-IDF environment
   - Load WiFi credentials from `.env`
   - Clean and build the project
   - Auto-detect the USB port
   - Flash firmware and start serial monitor

4. **Just build (no flashing):**
   ```bash
   ./scripts/build.sh
   ```

5. **Quick flash (skip build):**
   ```bash
   ./scripts/flash.sh --skip-build
   ```
   Useful for rapid iterations when the build hasn't changed.

### Manual Build Process

If you prefer manual control or need to troubleshoot:

1. **Set up ESP-IDF environment:**
   ```bash
   source /path/to/esp-idf/export.sh
   ```

2. **Set target chip:**
   ```bash
   idf.py set-target esp32s3
   ```

3. **Load WiFi credentials:**
   ```bash
   source scripts/setenv.sh
   ```

4. **Build:**
   ```bash
   idf.py build
   ```

5. **Flash and monitor:**
   ```bash
   idf.py -p /dev/cu.usbmodem* flash monitor
   ```
   
   **Port Detection:**
   - **Linux:** `/dev/ttyUSB0`, `/dev/ttyACM0`
   - **macOS:** `/dev/cu.usbserial-*` or `/dev/cu.usbmodem*`
   - **Windows:** `COM3`, `COM4`, etc.

   To exit the serial monitor, press `Ctrl+]`.

## Over-The-Air (OTA) Updates

This firmware is designed to support OTA updates from the local AI server. The `partitions.csv` file is configured for two application partitions (`app0` and `app1`).

* The `communications/ota_updater.c` module handles the client-side logic for checking for updates, downloading new firmware from the AI orchestration server (via HTTP), and rebooting into the new image.
* The AI orchestration system (in the `../ai-server/` directory) manages OTA updates through its device management component. Refer to the server's documentation for triggering OTA updates.

## Build Scripts

The `scripts/` directory contains automation tools for development:

* **`setenv.sh`**: Loads WiFi credentials from `../.env` and exports them as environment variables
* **`build.sh`**: Complete build automation - sets up ESP-IDF, loads credentials, cleans old builds, and compiles
* **`flash.sh`**: Build + flash + monitor with auto USB port detection
  * Use `--skip-build` flag for faster iterations when build hasn't changed

### Development Workflow

```bash
# Full build and flash (recommended)
./scripts/flash.sh

# Quick flash without rebuilding
./scripts/flash.sh --skip-build

# Build only (no flashing)
./scripts/build.sh
```