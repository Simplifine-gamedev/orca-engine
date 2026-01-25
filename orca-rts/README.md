# Orca RTS - Marquee Selection Demo

A real-time strategy (RTS) game demo showcasing marquee/drag selection for both units and buildings.

## Features

### Marquee Selection (ORC-108)
- ✅ Drag selection box to select multiple entities
- ✅ Works with both units (circles) and buildings (rectangles)
- ✅ Filters to only select player-owned entities
- ✅ Optional mixed selection (units + buildings together)
- ✅ Visual feedback with yellow selection indicators

### Game Elements
- **Units**: Circular entities that can move around the map
- **Buildings**: Rectangular structures (Headquarters, Barracks, Factory)
- **Player Ownership**: Blue/Green for player 1, Red for enemy player 2
- **Selection Box**: Green dashed rectangle while dragging

## Implementation Details

### Selection Logic (`src/App.tsx`)

The marquee selection system includes:

1. **Entity Detection**: Checks if entities (units/buildings) intersect with the selection box
2. **Player Filtering**: Only player-owned entities can be selected
3. **Mixed Selection**: Toggle to allow/prevent selecting units and buildings together
4. **Collision Detection**:
   - Units: Circle-rectangle intersection
   - Buildings: Rectangle-rectangle overlap

### Key Functions

- `isEntityInMarquee()`: Determines if an entity is within the selection box
- `handleMouseDown()`: Initiates marquee selection
- `handleMouseMove()`: Updates selection box size
- `handleMouseUp()`: Finalizes selection and updates entity states

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

```bash
cd orca-rts
npm install
```

### Development

```bash
npm start
```

Opens the game at [http://localhost:3000](http://localhost:3000)

### Build

```bash
npm run build
```

Creates optimized production build in the `build/` folder.

## Usage

1. **Select Units/Buildings**: Click and drag to create a selection box
2. **Toggle Mixed Selection**: Use the checkbox to allow/prevent mixed selection
3. **Visual Feedback**: Selected entities show yellow borders/outlines

### Controls
- **Mouse Drag**: Create marquee selection box
- **Checkbox**: Toggle mixed selection mode

### Entity Colors
- 🔵 Blue circles = Your units
- 🟢 Green rectangles = Your buildings  
- 🔴 Red circles = Enemy units
- 🔴 Red rectangles = Enemy buildings
- 🟡 Yellow border = Selected

## Architecture

```
orca-rts/
├── public/
│   └── index.html
├── src/
│   ├── App.tsx          # Main game logic and marquee selection
│   ├── App.css          # Styling
│   ├── index.tsx        # React entry point
│   └── index.css        # Global styles
├── package.json
├── tsconfig.json
└── README.md
```

## Future Enhancements

- [ ] Right-click commands (move, attack)
- [ ] Unit grouping and control groups
- [ ] Building construction
- [ ] Resource management
- [ ] Fog of war
- [ ] Pathfinding for unit movement
- [ ] Multiplayer support

## Linear Issue

This implements **ORC-108**: [Selection] Marquee/drag selection box for structures

- Extends marquee selection to include buildings ✅
- Filters to only select player-owned buildings ✅  
- Supports mixed selection (units + buildings) ✅

## License

Part of the Orca Engine project. See main repository for license details.
