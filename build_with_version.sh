#!/bin/bash
# Proper build script that ensures ORCA_VERSION gets to SCons

VERSION="${1:-0.01.9b45879b}"  # Use argument or default to latest GitHub version

echo "🏗️  BUILDING ORCA ENGINE WITH VERSION: $VERSION"
echo "================================================"

# Export ORCA_VERSION to environment
export ORCA_VERSION="$VERSION"

# Verify the environment variable is set
echo "📋 Environment check:"
echo "   ORCA_VERSION = $ORCA_VERSION"

# Clean the generated file to force regeneration
if [ -f "core/orca_version.gen.cpp" ]; then
    rm core/orca_version.gen.cpp
    echo "🗑️  Removed old orca_version.gen.cpp"
fi

echo ""
echo "🚀 Starting SCons build..."
echo "   The orca_version_builder should detect ORCA_VERSION=$ORCA_VERSION"

# Run SCons with explicit environment variable passing
env ORCA_VERSION="$ORCA_VERSION" scons platform=macos target=editor dev_build=yes vulkan=no -j4

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ BUILD COMPLETE!"
    echo ""
    echo "🔍 Checking embedded version:"
    if [ -f "core/orca_version.gen.cpp" ]; then
        EMBEDDED_VERSION=$(grep 'ORCA_VERSION_STRING = ' core/orca_version.gen.cpp | cut -d'"' -f2)
        echo "   Embedded: $EMBEDDED_VERSION"
        
        if [ "$EMBEDDED_VERSION" = "$VERSION" ]; then
            echo "   ✅ Version correctly embedded!"
        else
            echo "   ❌ Version mismatch! Expected: $VERSION, Got: $EMBEDDED_VERSION"
        fi
    else
        echo "   ❌ orca_version.gen.cpp not found!"
    fi
    
    echo ""
    echo "🧪 TO TEST:"
    echo "   ./bin/orca.macos.editor.dev.arm64"
    echo ""
    echo "   Look for console output:"
    echo "   'UpdateNotificationPopup: ✅ Version from compiled binary: $VERSION'"
    echo ""
    echo "   Expected behavior:"
    if [ "$VERSION" = "0.01.9b45879b" ]; then
        echo "   🔕 NO update notification (current version matches GitHub)"
    else
        echo "   🔔 SHOULD show update notification (version differs from GitHub latest)"
    fi
    
else
    echo "❌ BUILD FAILED"
fi
