# Orca RTS - Resource Selection Feature

This directory contains the Orca RTS game implementation with resource selection and info panel functionality.

## Feature: Resource Selection & Info Panel

Users can click on resources (gold mines, trees) to open an info panel showing:
- Resource type
- Amount remaining
- Workers assigned
- Gather rate
- Worker management controls

## Project Structure

```
src/
├── components/
│   └── Game.tsx              # Main game component with demo setup
├── resources/
│   ├── GoldMine.tsx          # Gold mine resource component
│   └── Tree.tsx              # Tree resource component
├── ui/
│   └── SelectionPanel.tsx    # Info panel UI for selected resources
├── store/
│   └── gameStore.ts          # Zustand state management
├── types/
│   └── resource.ts           # TypeScript type definitions
├── main.tsx                  # Entry point
├── index.html                # HTML template
├── package.json              # Dependencies
└── tsconfig.json             # TypeScript configuration
```

## Implementation Details

### State Management (`gameStore.ts`)
- Uses Zustand for lightweight state management
- Tracks selected entity and resources
- Provides actions for selection and worker management

### Resource Components
- **GoldMine.tsx**: Clickable gold mine with visual feedback
- **Tree.tsx**: Clickable tree resource with visual feedback
- Both show worker count badges and fill indicators

### Selection Panel (`SelectionPanel.tsx`)
- Displays when a resource is selected
- Shows resource details (type, amount, workers, gather rate)
- Interactive worker assignment controls (+/- buttons)
- Visual progress bars and worker icons

## Key Features Implemented

1. **Clickable Resources**: Resources respond to clicks and show selection state
2. **Info Panel**: Detailed panel appears at bottom of screen when resource selected
3. **Worker Management**: Add/remove workers with instant visual feedback
4. **Visual Feedback**: 
   - Selected resources have green border and glow
   - Resource fill indicators show remaining amount
   - Worker badges show assignment count
5. **Type Safety**: Full TypeScript implementation with proper types

## Testing

The `Game.tsx` component includes a demo setup with:
- 3 gold mines (various depletion levels)
- 2 trees
- Pre-assigned workers on some resources

Click any resource to see the info panel, then use the +/- buttons to manage workers.

## Dependencies

- React 18.2+
- Zustand 4.4+ (state management)
- TypeScript 5.0+

## Next Steps

Potential enhancements:
- Add more resource types (stone quarry, etc.)
- Implement actual resource gathering over time
- Add sound effects for clicks and worker assignment
- Multi-selection support
- Keyboard shortcuts (ESC to deselect, etc.)
