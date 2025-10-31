#!/bin/bash
# Update the Orca.app bundle with the latest dev binary

cd "$(dirname "$0")"

echo "Updating Orca.app with latest binary..."
cp bin/orca.macos.editor.dev.arm64 bin/Orca.app/Contents/MacOS/Orca
chmod +x bin/Orca.app/Contents/MacOS/Orca

echo "✅ App bundle updated successfully!"
echo "You can now run: open bin/Orca.app"


