@echo off
REM Windows Update Launcher for Orca Engine
REM This ensures proper argument passing during updates

echo Orca Engine Update: Launching with Project Manager mode...

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Look for Orca.exe in common locations
if exist "%SCRIPT_DIR%Orca.exe" (
    echo Found Orca.exe in script directory
    "%SCRIPT_DIR%Orca.exe" --project-manager
    goto :cleanup
)

if exist "%SCRIPT_DIR%orca.windows.editor.x86_64.exe" (
    echo Found orca.windows.editor.x86_64.exe in script directory  
    "%SCRIPT_DIR%orca.windows.editor.x86_64.exe" --project-manager
    goto :cleanup
)

REM Try common installation paths
for %%P in ("%ProgramFiles%\Orca" "%ProgramFiles(x86)%\Orca" "%LocalAppData%\Orca") do (
    if exist "%%P\Orca.exe" (
        echo Found Orca.exe at %%P
        "%%P\Orca.exe" --project-manager
        goto :cleanup
    )
)

echo ERROR: Could not find Orca.exe
echo Please run Orca.exe manually with --project-manager argument
pause

:cleanup
REM Self-delete this batch file
del "%~f0"
