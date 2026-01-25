# Orca RTS - Formation and Positioning Controls

A React-based RTS game demonstration featuring advanced unit formation and positioning controls, similar to Total War.

## Features

### Core Functionality
- **Multiple Unit Selection**: Select and control multiple units at once
- **Formation Types**: Line, Box, and Wedge formations
- **Spread Control**: Tight, Normal, and Loose spacing options
- **Facing Direction**: Drag to set unit facing direction (Total War style)
- **Path Visualization**: Toggle between individual paths and group path display

### Controls
- **Shift + Left Click**: Add a new unit at cursor position
- **Left Click**: Select a unit
- **Ctrl/Cmd + Click**: Toggle unit selection (multi-select)
- **Right Click**: Move selected units to target location
- **Right Click + Drag**: Set facing direction while moving units

## Installation

```bash
cd rts-game
npm install
```

## Development

```bash
npm run dev
```

Open your browser to `http://localhost:5173`

## Build

```bash
npm run build
```

## Project Structure

```
rts-game/
├── src/
│   ├── store/
│   │   └── gameStore.ts       # Zustand store for game state management
│   ├── units/
│   │   └── RTSUnit.tsx        # Unit component with path visualization
│   ├── utils/
│   │   └── formations.ts      # Formation calculation logic
│   ├── types/
│   │   └── index.ts           # TypeScript type definitions
│   ├── App.tsx                # Main application component
│   ├── App.css                # Application styles
│   └── main.tsx               # Application entry point
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Implementation Details

### Formation System
- **Line Formation**: Units align perpendicular to facing direction
- **Box Formation**: Units arrange in a grid pattern
- **Wedge Formation**: Units form a triangle, narrow at front, wide at back

### Movement Logic
- Units maintain formation when moving as a group
- Facing direction can be set via drag gesture
- Units smoothly animate to target positions

### State Management
- Uses Zustand for efficient state management
- All game logic contained in the store
- Reactive updates for UI components

## Technologies Used
- React 18
- TypeScript
- Zustand (state management)
- Vite (build tool)
- SVG for rendering

## License
MIT
