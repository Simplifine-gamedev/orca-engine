#!/bin/bash
# DMG Customization Script for Orca Engine
# Creates a beautifully customized DMG with drag arrow

set -e

DMG_NAME="$1"
VOLUME_NAME="$2"
SOURCE_DIR="$3"
BACKGROUND_IMG="misc/dmg-resources/dmg-background.png"

if [ -z "$DMG_NAME" ] || [ -z "$VOLUME_NAME" ] || [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 <dmg_name> <volume_name> <source_dir>"
    echo "Example: $0 'Orca.dmg' 'Orca Engine' 'dmg-staging'"
    exit 1
fi

echo "🎨 Creating customized DMG: $DMG_NAME"

# Create temporary DMG
TEMP_DMG="temp_$DMG_NAME"
hdiutil create -volname "$VOLUME_NAME" -srcfolder "$SOURCE_DIR" -ov -format UDRW "$TEMP_DMG"

# Mount the temporary DMG
echo "📦 Mounting temporary DMG..."
MOUNT_DIR=$(hdiutil attach -readwrite -noverify "$TEMP_DMG" | egrep '^/dev/' | sed 1q | awk '{print $3}')

echo "📁 Mounted at: $MOUNT_DIR"

# Copy background image to the DMG
echo "🖼️  Adding background image..."
mkdir -p "$MOUNT_DIR/.background"
cp "$BACKGROUND_IMG" "$MOUNT_DIR/.background/background.png"

# Set up the DMG view options using AppleScript
echo "⚙️  Customizing DMG layout..."
cat > /tmp/dmg_setup.applescript << 'EOF'
tell application "Finder"
    tell disk "Orca Engine"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {100, 100, 740, 500}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 128
        set background picture of viewOptions to file ".background:background.png"
        
        -- Position icons
        set position of item "Orca.app" of container window to {160, 200}
        set position of item "Applications" of container window to {480, 200}
        
        -- Hide background folder
        set the extension hidden of item ".background" to true
        
        close
        open
        update without registering applications
        delay 2
    end tell
end tell
EOF

# Run the AppleScript to customize the DMG
osascript /tmp/dmg_setup.applescript

# Sync and unmount
echo "💾 Syncing changes..."
sync
hdiutil detach "$MOUNT_DIR"

# Convert to compressed read-only DMG
echo "🗜️  Converting to final DMG..."
hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_NAME"

# Clean up
rm "$TEMP_DMG"
rm /tmp/dmg_setup.applescript

echo "✅ Customized DMG created: $DMG_NAME"
