# Windows-side setup for the NAILA audio client.
#
# Why this exists: WSLg's PulseAudio bridge is the single most flaky
# component in the WSL dev path. Running the audio client natively on
# Windows bypasses WSLg entirely — Windows audio (WASAPI) is rock solid,
# and the audio client connects to the WSL-hosted hub over localhost
# (WSL2 forwards ports both directions automatically).
#
# Usage (one-time setup):
#   pwsh .\scripts\win-audio-client-setup.ps1
#
# Then to launch the audio client:
#   pwsh .\scripts\win-audio-client-run.ps1
#
# Prerequisites:
#   - Python 3.12 installed on Windows (https://www.python.org/downloads/
#     or `winget install Python.Python.3.12`).
#   - This repo lives in WSL. The script reads it via the \\wsl.localhost\
#     UNC path; nothing needs to be cloned to Windows.

param(
    # Override these if your WSL distro name or username differs.
    [string]$WslDistro = "Ubuntu",
    [string]$WslUser = $env:USERNAME,
    [string]$VenvDir = "$env:USERPROFILE\.naila-audio-venv"
)

$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────────────────────────
# Locate the WSL repo
# ─────────────────────────────────────────────────────────────────────────────

$audioClientPath = "\\wsl.localhost\$WslDistro\home\$WslUser\naila\robo-naila\devices\audio-client"

if (-not (Test-Path $audioClientPath)) {
    Write-Host "Could not find the audio client at:" -ForegroundColor Red
    Write-Host "  $audioClientPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Override the path with -WslDistro and -WslUser, e.g.:"
    Write-Host "  pwsh .\scripts\win-audio-client-setup.ps1 -WslDistro Ubuntu-24.04 -WslUser yourname"
    exit 1
}
Write-Host "Found audio client at: $audioClientPath"

# ─────────────────────────────────────────────────────────────────────────────
# Find Python 3.12
# ─────────────────────────────────────────────────────────────────────────────

$python = (Get-Command py -ErrorAction SilentlyContinue)
if (-not $python) {
    Write-Host "Python launcher 'py' not found. Install Python 3.12:" -ForegroundColor Red
    Write-Host "  winget install Python.Python.3.12"
    exit 1
}

# Verify 3.12 is available via the launcher.
$pyVersion = (& py -3.12 --version 2>&1)
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.12 not available to 'py' launcher. Found:" -ForegroundColor Red
    & py --list
    Write-Host ""
    Write-Host "Install Python 3.12: winget install Python.Python.3.12"
    exit 1
}
Write-Host "Using Python: $pyVersion"

# ─────────────────────────────────────────────────────────────────────────────
# Create venv on the Windows side
# ─────────────────────────────────────────────────────────────────────────────
#
# We deliberately put the venv on the Windows filesystem (under $env:USERPROFILE)
# rather than alongside the WSL .venv. Two reasons:
#   1. Windows venvs and Linux venvs have different layouts and binaries — they
#      can't share a directory.
#   2. Python wheels installed on a Windows venv stored in the WSL filesystem
#      cross the 9p translation layer on every import — slow.

if (Test-Path $VenvDir) {
    Write-Host "Reusing existing venv at: $VenvDir"
} else {
    Write-Host "Creating Windows venv at: $VenvDir"
    & py -3.12 -m venv $VenvDir
}

$activate = "$VenvDir\Scripts\Activate.ps1"
$pip = "$VenvDir\Scripts\pip.exe"
$python = "$VenvDir\Scripts\python.exe"

# ─────────────────────────────────────────────────────────────────────────────
# Install audio-client in editable mode pointing at the WSL source
# ─────────────────────────────────────────────────────────────────────────────

Write-Host "Installing audio-client and dependencies..."
& $pip install --upgrade pip
& $pip install -e $audioClientPath

# ─────────────────────────────────────────────────────────────────────────────
# Sanity check: list audio devices
# ─────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Audio devices visible to PortAudio on Windows:" -ForegroundColor Cyan
& $python -c "import sounddevice as sd; print(sd.query_devices())"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "To launch the audio client:" -ForegroundColor Cyan
Write-Host "  pwsh .\scripts\win-audio-client-run.ps1"
Write-Host ""
Write-Host "Pick the device names you want for input/output from the list above"
Write-Host "and pass them via -InputDevice / -OutputDevice if the defaults don't"
Write-Host "match your hardware."
