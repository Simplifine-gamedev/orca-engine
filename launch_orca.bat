@echo off
REM Orca Engine Launcher - Forces Project Manager to open
REM This prevents auto-opening projects that might crash

cd /d "%~dp0"
start "" "orca.exe" --project-manager

