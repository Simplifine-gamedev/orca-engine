# Orca RTS - Unit Selection & Movement

A React/TypeScript RTS game with intuitive unit selection and movement controls.

## Features Implemented

### 1. Better Visual Feedback on Selection
- **Selection Rings**: Selected units display a glowing golden ring with pulse animation
- **Health Bars**: Color-coded health indicators above each unit
- **Movement Lines**: Visual lines showing unit movement paths
- **Target Indicators**: Animated markers at unit destination points
- **Hover Effects**: Units scale up when hovered for better interactivity

### 2. Clearer Move vs Attack Indicators
- **Dynamic Cursor**: Changes based on context (select/move/attack)
- **Move Indicator**: Green expanding circle animation at target location
- **Formation Movement**: Multiple units automatically arrange in formation
- **Movement Lines**: Dashed lines from units to their target positions
- **Visual Feedback**: Clear notifications for actions

### 3. Control Groups (Ctrl+1-9)
- **Assign Groups**: Press `Ctrl + 1-9` to assign selected units to a control group
- **Select Groups**: Press `1-9` to instantly select a control group
- **Additive Selection**: Hold `Shift` while pressing `1-9` to add control group to current selection
- **Visual Display**: Active control groups shown in UI panel

### 4. Tab to Cycle Through Selected Units
- **Cycle Selection**: Press `Tab` to cycle through currently selected units
- **Focus Management**: Helps manage individual units in large selections
- **Quick Access**: Rapidly switch between units without reselecting

### 5. Better Multi-Unit Selection Handling
- **Box Selection**: Click and drag to select multiple units
- **Additive Selection**: Hold `Shift` to add/remove units from selection
- **Clear Selection**: Press `Esc` to deselect all units
- **Formation Movement**: Units maintain proper spacing when moving as a group
- **Smart Targeting**: Right-click to move, or target enemies to attack

## Controls

### Mouse Controls
- **Left Click**: Select a single unit
- **Click + Drag**: Box select multiple units
- **Right Click**: Move selected units to location
- **Shift + Click**: Add/remove units from selection

### Keyboard Shortcuts
- **Ctrl + 1-9**: Assign selected units to control group
- **1-9**: Select control group
- **Shift + 1-9**: Add control group to current selection
- **Tab**: Cycle through selected units
- **Esc**: Clear all selections

## Unit Types

- **⚔️ Warrior**: Balanced fighter (Red)
- **🏹 Archer**: Fast ranged unit (Cyan)
- **🔮 Mage**: Slower magic unit (Purple)

## Installation & Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Technical Stack

- **React 18**: UI framework
- **TypeScript**: Type-safe development
- **Zustand**: Lightweight state management
- **Vite**: Fast build tool and dev server

## Project Structure

```
src/
├── App.tsx                 # Main game component with input handling
├── index.tsx              # React entry point
├── store/
│   └── gameStore.ts       # Zustand store for game state
├── units/
│   └── RTSUnit.tsx        # Unit component with visual feedback
└── types/
    └── unit.ts            # TypeScript type definitions
```

## Architecture Highlights

### State Management
- Centralized game state using Zustand
- Efficient unit tracking and selection management
- Real-time position updates with 60 FPS game loop

### Visual Feedback System
- Layered rendering with proper z-index management
- CSS animations for smooth visual transitions
- Dynamic styling based on unit state

### Input System
- Mouse event handling for selection and movement
- Keyboard shortcuts for advanced controls
- Modifier key support (Ctrl, Shift)

## Future Enhancements

- Attack commands and combat system
- Fog of war implementation
- Minimap display
- Unit production and resource management
- Multiplayer support
- AI opponents

## License

MIT License - Open source and free to use.
