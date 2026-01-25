# Orca RTS - Resource Info Panel

This implementation adds clickable resource functionality to the Orca RTS game, allowing players to view detailed information about resources like gold mines and trees.

## Features Implemented

### 1. Selectable Resources
- Click on any resource (gold mine or tree) to select it
- Selected resources show visual feedback with a white border and glow effect
- Click on the background to deselect

### 2. Info Panel
When a resource is selected, the info panel displays:
- **Resource Type**: Shows the type with an icon (⛏️ for gold mines, 🌲 for trees)
- **Amount Remaining**: Current vs maximum capacity with a progress bar
- **Workers Assigned**: Number of workers currently gathering from the resource
- **Gather Rate**: Resources gathered per second (rate × workers)

### 3. Worker Management
- **Add Worker**: Increase workers assigned to the selected resource
- **Remove Worker**: Decrease workers (disabled when none assigned)
- Worker count is displayed as a badge on resources with active workers

## Files Modified/Created

```
src/
├── types/
│   └── index.ts           # TypeScript interfaces for Resource and GameState
├── store/
│   └── gameStore.ts       # Zustand state management with selection logic
├── resources/
│   └── GoldMine.tsx       # Resource component with click handlers
├── ui/
│   └── SelectionPanel.tsx # Info panel UI component
├── Game.tsx               # Main game component
└── index.tsx              # Application entry point
```

## How to Run

### Development Mode
```bash
npm install
npm run dev
```
Visit http://localhost:3000 to play the game.

### Production Build
```bash
npm run build
npm run preview
```

## Game Interactions

1. **Select a Resource**: Click on any gold mine or tree on the map
2. **View Info**: The selection panel appears in the bottom-right corner
3. **Manage Workers**: Use the + and - buttons to adjust worker assignments
4. **Deselect**: Click on empty space to close the info panel

## Technical Details

- **State Management**: Zustand for lightweight, type-safe state management
- **Styling**: Inline styles with React for rapid development and no CSS dependencies
- **TypeScript**: Full type safety across all components
- **Build Tool**: Vite for fast development and optimized production builds

## Initial Resources

The game starts with three resources:
- **Gold Mine 1**: 5000/5000 gold, 0 workers
- **Gold Mine 2**: 3500/5000 gold, 2 workers
- **Tree 1**: 500/500 wood, 1 worker

## Future Enhancements

Potential improvements:
- Add more resource types
- Implement actual resource gathering mechanics
- Add unit selection and pathfinding
- Implement building placement
- Add multiplayer support
