#!/bin/bash
# Test script to demonstrate building Orca with embedded version

echo "🔧 ORCA ENGINE - TEST BUILD WITH EMBEDDED VERSION"
echo "=================================================="

# Get current git info
CURRENT_SHA=$(git rev-parse --short=8 HEAD)
echo "📍 Current git SHA: $CURRENT_SHA"

# Set test version
export ORCA_VERSION="0.01.$CURRENT_SHA"
echo "📦 Building with version: $ORCA_VERSION"

# Build Godot (adjust platform as needed)
echo ""
echo "🏗️  Starting SCons build..."
echo "   Platform: macos"
echo "   Target: editor"
echo "   Version: $ORCA_VERSION (will be embedded)"

# Uncomment to actually build:
# scons platform=macos target=editor -j8

# For now, just show what would happen
echo ""
echo "✨ During build, SCons will:"
echo "   1. Run orca_version_builder() in core_builders.py"
echo "   2. Generate core/orca_version.gen.cpp with:"
echo "      const char *const ORCA_VERSION_STRING = \"$ORCA_VERSION\";"
echo "   3. Compile it into the binary"
echo ""
echo "📱 When users run the binary:"
echo "   - Checks GitHub API for latest release"
echo "   - Compares embedded $ORCA_VERSION with GitHub tag"
echo "   - Only shows notification if different"
echo ""
echo "🧪 TO TEST:"
echo "   1. Actually run the build: scons platform=macos target=editor"
echo "   2. Launch: ./bin/godot.macos.editor.arm64"
echo "   3. Check console for: 'Version from compiled binary: $ORCA_VERSION'"
echo "   4. Should NOT show update notification (versions match)"

