# Builds dist\VoiceBridge-SelfHosted-Agent.exe (one file, no console window) and copies it to the web app's downloads.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\pyinstaller.exe --noconfirm --onefile --noconsole --name VoiceBridge-SelfHosted-Agent `
    --hidden-import pystray._win32 `
    voicebridge_agent.py
New-Item -ItemType Directory -Force ..\frontend\public\downloads | Out-Null
Copy-Item dist\VoiceBridge-SelfHosted-Agent.exe ..\frontend\public\downloads\VoiceBridge-SelfHosted-Agent.exe -Force
Write-Host "Built dist\VoiceBridge-SelfHosted-Agent.exe and copied it to frontend\public\downloads"
