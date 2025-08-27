#!/bin/bash

# Script to update the Orca binary after compilation

echo "Updating Orca binary with the newly compiled version..."

# Check if the compiled binary exists
if [ -f "bin/godot.macos.editor.dev.arm64" ]; then
    # Backup the current binary
    cp bin/Orca.app/Contents/MacOS/Orca bin/Orca.app/Contents/MacOS/Orca.backup_$(date +%Y%m%d_%H%M%S)
    
    # Copy the new binary
    cp bin/godot.macos.editor.dev.arm64 bin/Orca.app/Contents/MacOS/Orca
    
    # Also update the hidden legacy file for consistency
    cp bin/godot.macos.editor.dev.arm64 bin/.legacy_orca.macos.editor.dev.arm64
    
    echo "✅ Orca binary updated successfully!"
    echo "The dock icon transparency fix should now be active."
    echo ""
    echo "To test:"
    echo "1. Run: bin/Orca"
    echo "2. Open the project manager - check dock icon (should be transparent)"
    echo "3. Open a project - check dock icon (should stay transparent)"
else
    echo "❌ Error: Compiled binary not found at bin/godot.macos.editor.dev.arm64"
    echo "Please wait for compilation to complete."
fi
