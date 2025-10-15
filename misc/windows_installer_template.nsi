# Orca Engine Windows Installer (NSIS Template)
# This ensures proper argument passing for auto-updates

!define PRODUCT_NAME "Orca Engine"
!define PRODUCT_VERSION "1.0.0"
!define PRODUCT_PUBLISHER "Your Company"

# Installer settings
Name "${PRODUCT_NAME}"
OutFile "Orca-Setup.exe"
InstallDir "$PROGRAMFILES\Orca"

# Pages
Page directory
Page instfiles

Section "Install"
    SetOutPath "$INSTDIR"
    
    # Copy files
    File /r "path\to\orca\binary\*.*"
    
    # CRITICAL: Create shortcuts with proper arguments
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    
    # Editor shortcut (default)
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Orca Engine.lnk" \
        "$INSTDIR\Orca.exe" \
        "--project-manager" \
        "$INSTDIR\Orca.exe" \
        0 \
        SW_SHOWNORMAL \
        "" \
        "Launch Orca Engine"
    
    # Desktop shortcut
    CreateShortCut "$DESKTOP\Orca Engine.lnk" \
        "$INSTDIR\Orca.exe" \
        "--project-manager" \
        "$INSTDIR\Orca.exe" \
        0
    
    # CRITICAL FOR AUTO-UPDATE: Add registry entry so updated exe knows to launch in editor mode
    WriteRegStr HKLM "Software\Orca" "DefaultMode" "project-manager"
    WriteRegStr HKLM "Software\Orca" "InstallPath" "$INSTDIR"
    
    # Uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\*.*"
    RMDir /r "$INSTDIR"
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.*"
    RMDir "$SMPROGRAMS\${PRODUCT_NAME}"
    Delete "$DESKTOP\Orca Engine.lnk"
    DeleteRegKey HKLM "Software\Orca"
SectionEnd
