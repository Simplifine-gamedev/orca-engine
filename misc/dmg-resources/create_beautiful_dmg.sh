#!/bin/bash
# Professional DMG creation using create-dmg library
# This creates beautiful DMGs like the ones used by major Mac apps

set -e

APP_NAME="$1"
DMG_NAME="$2"
SOURCE_DIR="$3"

if [ -z "$APP_NAME" ] || [ -z "$DMG_NAME" ] || [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 <app_name> <dmg_name> <source_dir>"
    echo "Example: $0 'Orca.app' 'Orca (Mac).dmg' 'dmg-staging'"
    exit 1
fi

echo "🎨 Creating beautiful DMG using create-dmg library..."

# Use create-dmg to build a professional DMG
create-dmg \
  --volname "Orca Engine" \
  --volicon "misc/dmg-resources/volume-icon.icns" \
  --background "misc/dmg-resources/dmg-background.png" \
  --window-pos 200 120 \
  --window-size 640 400 \
  --icon-size 128 \
  --icon "$APP_NAME" 160 200 \
  --hide-extension "$APP_NAME" \
  --app-drop-link 480 200 \
  --skip-jenkins \
  "$DMG_NAME" \
  "$SOURCE_DIR"

echo "✅ Beautiful DMG created: $DMG_NAME"
