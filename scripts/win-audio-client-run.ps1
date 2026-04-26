# Launch the NAILA audio client on Windows, pointing at the hub running in WSL.
#
# Usage (after setup):
#   pwsh .\scripts\win-audio-client-run.ps1
#
# Common overrides:
#   pwsh .\scripts\win-audio-client-run.ps1 `
#     -HubUrl http://localhost:8080 `
#     -DeviceId dev-windows `
#     -InputDevice "Microphone (Yeti)" `
#     -OutputDevice "Speakers (Realtek)"
#
# Device names: run the setup script (or `python -c "import sounddevice as sd;
# print(sd.query_devices())"` from the venv) to see what PortAudio sees on
# this machine. Names are stable across reboots; indices aren't.

param(
    [string]$VenvDir = "$env:USERPROFILE\.naila-audio-venv",
    [string]$HubUrl = "http://localhost:8080",
    [string]$DeviceId = "dev-windows",
    [string]$InputDevice = "",
    [string]$OutputDevice = ""
)

$ErrorActionPreference = "Stop"

$python = "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Venv not found at $VenvDir." -ForegroundColor Red
    Write-Host "Run setup first: pwsh .\scripts\win-audio-client-setup.ps1"
    exit 1
}

$cliArgs = @(
    "-m", "audio_client",
    "--hub-url", $HubUrl,
    "--device-id", $DeviceId
)
if ($InputDevice) { $cliArgs += @("--input-device", $InputDevice) }
if ($OutputDevice) { $cliArgs += @("--output-device", $OutputDevice) }

Write-Host "Launching audio client (Ctrl+C to stop)..." -ForegroundColor Cyan
Write-Host ("  hub-url: {0}" -f $HubUrl)
Write-Host ("  device-id: {0}" -f $DeviceId)
if ($InputDevice) { Write-Host ("  input: {0}" -f $InputDevice) } else { Write-Host "  input: (default)" }
if ($OutputDevice) { Write-Host ("  output: {0}" -f $OutputDevice) } else { Write-Host "  output: (default)" }
Write-Host ""

& $python @cliArgs
