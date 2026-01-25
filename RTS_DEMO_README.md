# RTS Formation Control Demo

This demo implements formation and positioning control for multiple units in an RTS-style game interface.

## Features Implemented

### 1. Formation Types
- **None**: Units move in a simple grid formation
- **Line**: Units form a horizontal line perpendicular to facing direction
- **Box**: Units arrange in a rectangular grid
- **Wedge**: Units form a triangular wedge pointing in the facing direction

### 2. Spread Control
- **Tight**: Units are closer together (0.5x spacing)
- **Normal**: Default spacing (1.0x)
- **Loose**: Units are more spread out (2.0x spacing)

### 3. Facing Direction Control
Hold **Shift + Right Click and Drag** to set the facing direction for your formation (like Total War games):
- Click and hold Shift + Right mouse button on the target location
- Drag in the direction you want units to face
- Release to command units to move in formation with that facing

### 4. Path Visualization
- **Individual Paths**: Toggle to show/hide path lines for each unit
- **Group Path**: Shows a single thicker path line representing the group movement

### 5. Controls
- **Left Click**: Select a single unit
- **Click and Drag**: Box select multiple units
- **Shift + Left Click**: Add unit to selection
- **Right Click**: Command selected units to move
- **Shift + Right Click + Drag**: Set facing direction and move

## Files Structure

```
src/
├── types/
│   └── index.ts              # TypeScript type definitions
├── store/
│   └── gameStore.ts          # Movement logic and formation calculations
├── units/
│   └── RTSUnit.tsx           # Unit rendering and path visualization
├── App.tsx                   # Main app with input handling
├── App.css                   # Styling
├── index.tsx                 # React entry point
└── index.html                # HTML entry point
```

## How It Works

### Formation Calculations
The `gameStore.ts` implements different formation algorithms:

- **Line Formation**: Places units perpendicular to the facing direction
- **Box Formation**: Arranges units in a grid, then rotates the grid
- **Wedge Formation**: Creates rows of increasing width pointing forward

### Movement System
Units smoothly interpolate to their target positions at a constant speed. The animation loop in `App.tsx` updates unit positions every frame.

### Input Handling
- Mouse events are converted to SVG coordinates
- Selection is managed through a combination of direct clicks and drag selection
- Formation direction is calculated from the drag vector angle

## Running the Demo

To run this demo, you would typically:

```bash
# Install dependencies (if package.json is set up)
npm install

# Run development server
npm run dev
```

However, this is a demonstration implementation for the Orca RTS project. The files can be integrated into the main Orca Engine or run as a standalone web application.

## Future Enhancements

Potential improvements could include:
- Pathfinding around obstacles
- Unit collision avoidance
- More formation types (circle, staggered, etc.)
- Formation rotation without moving
- Unit behavior states (aggressive, defensive, etc.)
- Minimap display
- Keyboard shortcuts for formation switching
