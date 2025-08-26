#!/bin/bash
# Load WiFi credentials from .env file - DO NOT COMMIT THIS FILE

# Check if .env exists in project root
ENV_FILE="../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Create a .env file in the project root with:"
    echo "WIFI_SSID=your_network_name"
    echo "WIFI_PASSWORD=your_password"
    exit 1
fi

# Source the .env file directly
set -a  # automatically export all variables
source "$ENV_FILE"
set +a

# Map to ESP-IDF expected variable names
if [ -n "$WIFI_SSID" ]; then
    export CONFIG_EXAMPLE_WIFI_SSID="$WIFI_SSID"
    echo "Set CONFIG_EXAMPLE_WIFI_SSID=$WIFI_SSID"
fi

if [ -n "$WIFI_PASSWORD" ]; then
    export CONFIG_EXAMPLE_WIFI_PASSWORD="$WIFI_PASSWORD"
    echo "Set CONFIG_EXAMPLE_WIFI_PASSWORD=***"
fi

if [[ -z "$CONFIG_EXAMPLE_WIFI_SSID" ]] || [[ -z "$CONFIG_EXAMPLE_WIFI_PASSWORD" ]]; then
    echo "ERROR: Missing required WiFi credentials in .env file"
fi

echo "WiFi environment variables loaded from .env"