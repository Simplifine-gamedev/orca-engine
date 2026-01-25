# Orca RTS - Unit Selection System

## Overview

This is a demonstration of the marquee selection box feature for the Orca RTS game that allows selecting both friendly and enemy units.

## Features Implemented (ORC-109)

### 1. Marquee Selection for All Units
- Click and drag to create a selection box
- Works for friendly, enemy, and neutral units
- Hold Shift to add to existing selection

### 2. Visual Distinction
- **Green** selection ring and box color for friendly units
- **Red** selection ring and box color for enemy units  
- **Yellow** selection ring for neutral units
- **Yellow** box color when selecting mixed unit types

### 3. Enemy Unit Information Display
- Enemy units show their info (name, health, position) when selected
- Clear warning message indicates enemy units cannot receive commands
- Only friendly units show command buttons (Move, Attack, Stop)

## Controls

- **Left Click**: Select a single unit
- **Click + Drag**: Create marquee selection box
- **Shift + Click**: Add/remove unit from selection
- **Shift + Drag**: Add units in box to selection

## Visual Indicators

| Team | Unit Color | Selection Ring | Selection Box |
|------|-----------|----------------|---------------|
| Friendly | Blue | Green | Green |
| Enemy | Red | Red | Red |
| Neutral | Gold | Yellow | Green |
| Mixed | - | - | Yellow |

## Project Structure

```
src/
├── App.tsx              # Main application with selection logic
├── App.css              # Styles
├── types.ts             # TypeScript interfaces and types
├── components/
│   ├── GameCanvas.tsx   # Canvas rendering component
│   └── UnitInfo.tsx     # Unit information sidebar
└── utils/
    └── selection.ts     # Selection utility functions
```

## Installation

```bash
cd src
npm install
```

## Development

```bash
npm run dev
```

Open http://localhost:3000 to view the demo.

## Building

```bash
npm run build
```

## Implementation Notes

- Units are rendered on HTML5 Canvas for performance
- Selection state is managed in React
- Visual feedback updates in real-time as you drag the selection box
- Health bars show unit status
- Grid background helps with spatial awareness
