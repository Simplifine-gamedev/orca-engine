#!/bin/bash
# Quick test script for 2D Animation System
# Run this to verify the complete integration

echo "🎬 2D Animation System - Quick Test"
echo "===================================="
echo ""

# Check if animation server is running
echo "1. Checking animation server (port 8001)..."
if curl -s http://127.0.0.1:8001/ > /dev/null 2>&1; then
    echo "   ✅ Animation server is running"
else
    echo "   ❌ Animation server NOT running!"
    echo "   Start it with: cd sprite_sheet_gen && python animation_server.py --workers 4 --port 8001"
    exit 1
fi

# Check if main backend is running
echo ""
echo "2. Checking main backend (port 5050)..."
if curl -s http://127.0.0.1:5050/health > /dev/null 2>&1; then
    echo "   ✅ Main backend is running"
else
    echo "   ❌ Main backend NOT running!"
    echo "   Start it with: export DEV_MODE=true && python app.py"
    exit 1
fi

# Test the 2d_animation_manager tool registration
echo ""
echo "3. Testing tool registration..."
echo "   Check backend logs for '2d_animation_manager' in tool list"

# Print environment variables
echo ""
echo "4. Environment Configuration:"
echo "   DEV_MODE: ${DEV_MODE:-not set}"
echo "   ANIMATION_SERVER_URL: ${ANIMATION_SERVER_URL:-not set (will use localhost)}"

echo ""
echo "✅ System is ready for testing!"
echo ""
echo "📝 Test in Godot Chat:"
echo "   1. 'Create a pixel-art robot with idle and walk animations'"
echo "   2. 'Check my animation status'"
echo "   3. 'Show all my sprite animations'"
echo "   4. 'Make #1 faster'"
echo ""
echo "🔍 Watch backend terminal for:"
echo "   - 2D_ANIM_CREATE logs"
echo "   - Image isolation process"
echo "   - Animation server communication"
echo "   - Cache updates"


