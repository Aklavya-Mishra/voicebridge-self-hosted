# Builds dist\VoiceBridgeAgent.exe (one file, no console window) and copies it to the web app's downloads.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\pyinstaller.exe --noconfirm --onefile --noconsole --name VoiceBridgeAgent `
    --hidden-import pystray._win32 `
    voicebridge_agent.py
New-Item -ItemType Directory -Force ..\frontend\public\downloads | Out-Null
Copy-Item dist\VoiceBridgeAgent.exe ..\frontend\public\downloads\VoiceBridgeAgent.exe -Force
Write-Host "Built dist\VoiceBridgeAgent.exe and copied it to frontend\public\downloads"
