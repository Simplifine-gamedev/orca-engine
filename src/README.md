# Orca RTS Game - Building Placement with Rotation

A real-time strategy game feature demonstrating building placement with rotation capabilities.

## Features

- **Building Selection**: Choose from multiple building types (Barracks, Tower, Farm, Factory)
- **Rotation Control**: Rotate buildings before placement using:
  - **R key**: Rotates the building ghost 90 degrees
  - **Mouse scroll wheel**: Alternative rotation control
- **Ghost Preview**: See building placement and rotation before confirming
- **Visual Feedback**: Buildings display their current rotation angle
- **Grid-based Placement**: Snap to grid for organized base building

## Building Rotation

When placing a building, you can rotate it in 90-degree increments (0°, 90°, 180°, 270°). The rotation is:
- Shown in the ghost preview before placement
- Saved when the building is placed
- Displayed on the placed building

## How to Use

1. **Select a building type** from the toolbar at the top
2. **Move your mouse** over the map to position the building
3. **Press R or scroll the mouse wheel** to rotate the building
4. **Click** to place the building at the current position and rotation
5. **Click on a placed building** to remove it
6. **Press ESC** to cancel building placement

## Controls

- **R key**: Rotate building 90 degrees clockwise
- **Mouse wheel**: Rotate building 90 degrees clockwise
- **Left click**: Place building or remove placed building
- **ESC**: Cancel building selection
- **Mouse move**: Position building ghost

## Project Structure

```
src/
├── App.tsx                    # Main app component with input handling
├── buildings/
│   └── Building.tsx          # Building component with rotation visual support
├── store/
│   └── gameStore.ts          # Game state management with Zustand
├── index.tsx                 # App entry point
├── index.html                # HTML template
├── vite.config.ts            # Vite configuration
├── tsconfig.json             # TypeScript configuration
└── package.json              # Dependencies
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

This starts the development server at http://localhost:3000

## Build

```bash
npm run build
```

This creates an optimized production build in the `dist` folder.

## Implementation Details

### Rotation System

The rotation is implemented using:
1. **State Management**: `ghostRotation` and `building.rotation` track rotation angle
2. **Visual Transform**: CSS `transform: rotate()` applies the rotation
3. **Dimension Swapping**: Width/height are swapped for 90° and 270° rotations
4. **Input Handling**: Keyboard and mouse wheel events trigger rotation

### Building Types

Each building type has:
- `id`: Unique identifier
- `name`: Display name
- `width`: Grid width (in cells)
- `height`: Grid height (in cells)
- `color`: Visual color
- `rotation`: Rotation angle (0, 90, 180, 270)

### Placed Buildings

Each placed building stores:
- `id`: Unique instance identifier
- `type`: Building type reference
- `x, y`: Grid position
- `rotation`: Rotation angle when placed

## Technologies Used

- **React 18**: UI framework
- **TypeScript**: Type-safe development
- **Zustand**: State management
- **Vite**: Build tool and dev server
