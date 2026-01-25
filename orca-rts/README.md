# Orca RTS - Real-Time Strategy Game

A web-based RTS game with building placement and rotation features.

## Features

- Building placement system
- Real-time building rotation (press R key)
- Ghost preview of buildings before placement
- Multiple building types: Barracks, Factory, Power Plant, Mine
- Visual rotation indicator

## Controls

- **R key**: Rotate building during placement
- **ESC key**: Cancel building placement
- **Left Click**: Place building
- **Mouse Move**: Preview building location

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The game will be available at `http://localhost:3000`

### Build

```bash
npm run build
```

## Technology Stack

- React 18
- TypeScript
- Zustand (State Management)
- Vite (Build Tool)

## Implementation Details

### Building Rotation

Buildings can be rotated in 90-degree increments (0°, 90°, 180°, 270°). The rotation is:
- Controlled by pressing the R key while placing a building
- Visually previewed in the ghost building display
- Preserved when the building is placed
- Displayed using CSS transforms for smooth rendering

### File Structure

- `src/store/gameStore.ts` - Game state management using Zustand
- `src/buildings/Building.tsx` - Building component with rotation rendering
- `src/App.tsx` - Main application with input handling

## User Feedback

Implementation based on user feedback:
- Gaudio: "tried to rotate the buildings" - Now supported with R key!
