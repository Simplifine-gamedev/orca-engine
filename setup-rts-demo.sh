#!/bin/bash

# Setup script for RTS Formation Demo

echo "Setting up RTS Formation Control Demo..."
echo ""

# Copy package.json to correct location
if [ -f "rts-demo-package.json" ]; then
    echo "Copying package.json..."
    cp rts-demo-package.json package-rts.json
    echo "Created package-rts.json"
fi

echo ""
echo "To run the RTS demo:"
echo "1. Install dependencies:"
echo "   npm install --prefix . (using package-rts.json)"
echo ""
echo "2. Start development server:"
echo "   npm run dev"
echo ""
echo "3. Open browser to http://localhost:3000"
echo ""
echo "Demo features:"
echo "  - Formation types: Line, Box, Wedge"
echo "  - Spread control: Tight, Normal, Loose"
echo "  - Drag to set facing direction (Shift + Right Click)"
echo "  - Toggle individual/group path visualization"
echo ""
echo "Controls:"
echo "  Left Click: Select unit"
echo "  Drag: Box select"
echo "  Shift + Click: Add to selection"
echo "  Right Click: Move units"
echo "  Shift + Right Click + Drag: Set facing direction"
