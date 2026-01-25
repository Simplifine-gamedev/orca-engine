#!/bin/bash
# ORC-112 Demo Runner Script

set -e

echo "========================================"
echo "ORC-112: Building Preview Fix Demo"
echo "========================================"
echo ""

# Find the Godot editor binary
GODOT_BIN=""
for pattern in "godot.*.editor.*" "godot" "Godot"; do
    found=$(find /workspace/bin -maxdepth 1 -name "$pattern" -executable 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        GODOT_BIN="$found"
        break
    fi
done

if [ -z "$GODOT_BIN" ]; then
    echo "❌ ERROR: Could not find Godot editor binary in /workspace/bin/"
    echo ""
    echo "Please build Orca Engine first:"
    echo "  scons platform=linuxbsd target=editor dev_build=yes vulkan=no -j\$(nproc)"
    exit 1
fi

echo "✅ Found Godot: $GODOT_BIN"
echo ""
echo "Starting demo..."
echo ""
echo "Controls:"
echo "  1-4: Switch Faction (Human, Dwarf, Elf, Undead)"
echo "  SPACE: Change Building Type"
echo "  CLICK: Place Building"
echo "  ESC: Quit"
echo ""
echo "Expected Behavior:"
echo "  ✅ Each faction shows different colored buildings"
echo "  ✅ Preview updates immediately when switching factions"
echo "  ✅ Placed buildings match preview"
echo ""
echo "========================================"
echo ""

# Run the demo
"$GODOT_BIN" --path /workspace/demo/rts-building-preview main.tscn

echo ""
echo "Demo closed."
