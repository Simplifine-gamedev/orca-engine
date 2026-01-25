# Orca RTS - Unit Training Queue System

Implementation of SHIFT+click to queue multiple unit training for Orca RTS game.

## Features

### 1. SHIFT Key Detection
- Click train button: Queue 1 unit
- SHIFT+Click train button: Queue 5 units
- Visual feedback and tooltips

### 2. Queue Management
- Display queue count badge on buildings
- Show detailed queue in SelectionPanel
- Progress bar for currently training unit
- Cancel individual units from queue

### 3. UI Components

#### Building Component (`src/buildings/Building.tsx`)
- Renders building on game map
- Shows queue count badge
- Train buttons appear when selected
- Detects SHIFT key state on click

#### SelectionPanel Component (`src/ui/SelectionPanel.tsx`)
- Shows selected building details
- Displays full training queue
- Progress indicator for active training
- Cancel buttons for each queued unit
- Helpful SHIFT+click tip

#### Game Store (`src/store/gameStore.ts`)
- Zustand state management
- Building and unit queue data structures
- Actions: trainUnit, cancelUnit, selectBuilding
- Progress tracking and unit completion

## Usage

```tsx
import { App } from './App';

// The App component demonstrates the complete system:
// 1. Initialize buildings
// 2. Render Building components
// 3. Show SelectionPanel for queue management
```

## Implementation Details

### Training Units
```typescript
// Train 1 unit (normal click)
trainUnit('barracks-1', 'Soldier', 1);

// Train 5 units (SHIFT+click)
trainUnit('barracks-1', 'Soldier', 5);
```

### Queue Structure
```typescript
interface UnitQueueItem {
  id: string;
  unitType: string;
  timestamp: number;
  progress: number; // 0-100
}
```

### SHIFT Key Detection
```typescript
const handleTrainUnit = (unitType: string, event: React.MouseEvent) => {
  const count = event.shiftKey ? 5 : 1;
  trainUnit(buildingId, unitType, count);
};
```

## Key Files

- `src/store/gameStore.ts` - State management with Zustand
- `src/buildings/Building.tsx` - Building component with SHIFT+click
- `src/ui/SelectionPanel.tsx` - Queue display UI
- `src/App.tsx` - Demo application

## Dependencies

- React 18+
- TypeScript 5+
- Zustand 4+ (state management)

## Testing

1. Click on a building to select it
2. Click "Train [Unit]" button - 1 unit queued
3. Hold SHIFT and click "Train [Unit]" - 5 units queued
4. Check queue count badge on building
5. View detailed queue in SelectionPanel
6. Click "Cancel" to remove units from queue

## Integration

To integrate with Orca Engine's Godot-based game:

1. Use GDScript to detect SHIFT key state
2. Call training functions with count parameter
3. Update UI through Godot's UI system or embedded web view
4. Bridge state between Godot and React if using hybrid approach

## Future Enhancements

- Configurable queue size (not just 5)
- CTRL+click for custom count
- Queue time estimates
- Resource costs display
- Sound effects for queue actions
- Hotkeys for unit types
