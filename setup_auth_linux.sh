#!/bin/bash
# Setup script for Orca Engine authentication on Linux

set -e

echo "==================================="
echo "Orca Engine Authentication Setup"
echo "==================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: This script is for Linux only"
    exit 1
fi

# Find the desktop file
DESKTOP_FILE="platform/linuxbsd/orca.desktop"

if [ ! -f "$DESKTOP_FILE" ]; then
    echo "Error: Could not find $DESKTOP_FILE"
    echo "Please run this script from the Orca Engine root directory"
    exit 1
fi

# Create applications directory if it doesn't exist
APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"

# Copy desktop file
echo "Installing desktop file..."
cp "$DESKTOP_FILE" "$APPS_DIR/"
echo "✓ Desktop file installed to $APPS_DIR/orca.desktop"

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    echo "Updating desktop database..."
    update-desktop-database "$APPS_DIR"
    echo "✓ Desktop database updated"
else
    echo "⚠ update-desktop-database not found, skipping"
fi

# Register URL scheme
if command -v xdg-mime &> /dev/null; then
    echo "Registering orca:// URL scheme..."
    xdg-mime default orca.desktop x-scheme-handler/orca
    echo "✓ URL scheme registered"
else
    echo "⚠ xdg-mime not found, URL scheme not registered"
    echo "   Please install xdg-utils package"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "You can now test the URL scheme with:"
echo "  xdg-open 'orca://auth?access_token=test&refresh_token=test&user_id=test'"
echo ""
echo "Note: Make sure to update the Exec path in $APPS_DIR/orca.desktop"
echo "      to point to your Orca binary location"
echo ""






