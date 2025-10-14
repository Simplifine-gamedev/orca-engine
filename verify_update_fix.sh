#!/bin/bash
# Quick verification that the update system is properly configured

echo "🔍 VERIFYING ORCA UPDATE SYSTEM FIX"
echo "===================================="

# Check if key files exist
echo ""
echo "📁 Checking generated files:"

if [ -f "core/orca_version.h" ]; then
    echo "✅ core/orca_version.h"
else
    echo "❌ core/orca_version.h - MISSING"
fi

if [ -f "core/orca_version.gen.cpp" ]; then
    echo "✅ core/orca_version.gen.cpp"
    VERSION_IN_FILE=$(grep 'ORCA_VERSION_STRING = ' core/orca_version.gen.cpp | cut -d'"' -f2)
    echo "   Embedded version: $VERSION_IN_FILE"
else
    echo "⚠️  core/orca_version.gen.cpp - Will be generated during SCons build"
fi

# Check SCons integration
echo ""
echo "🏗️  Checking SCons integration:"
if grep -q "orca_version_builder" core/SCsub; then
    echo "✅ core/SCsub has orca_version integration"
else
    echo "❌ core/SCsub - MISSING orca_version integration"
fi

if grep -q "def orca_version_builder" core/core_builders.py; then
    echo "✅ core/core_builders.py has orca_version_builder function"
else
    echo "❌ core/core_builders.py - MISSING orca_version_builder"
fi

# Check frontend integration
echo ""
echo "📱 Checking frontend integration:"
if grep -q "orca_version.h" editor/update/update_notification_popup.cpp; then
    echo "✅ update_notification_popup.cpp includes orca_version.h"
else
    echo "❌ update_notification_popup.cpp - MISSING orca_version.h include"
fi

if grep -q "ORCA_VERSION_STRING" editor/update/update_notification_popup.cpp; then
    echo "✅ update_notification_popup.cpp uses ORCA_VERSION_STRING"
else  
    echo "❌ update_notification_popup.cpp - NOT using ORCA_VERSION_STRING"
fi

# Check version comparison logic
echo ""
echo "🔄 Checking version comparison logic:"
if grep -q "remote_version == current_version" editor/update/update_notification_popup.cpp; then
    echo "✅ Simple version comparison (no spam logic)"
else
    echo "⚠️  Version comparison may need review"
fi

# Summary
echo ""
echo "════════════════════════════════════"
echo "📊 VERIFICATION COMPLETE"
echo "════════════════════════════════════"
echo ""
echo "🎯 NEXT STEPS:"
echo "1. Build with: export ORCA_VERSION='0.01.test' && scons platform=macos target=editor"
echo "2. Check embedded version: strings ./bin/godot.macos.editor.arm64 | grep '0\.01\.test'"
echo "3. Launch and verify no spam notifications"
echo ""
echo "📖 Full documentation: UPDATE_FIX_SUMMARY.md"

