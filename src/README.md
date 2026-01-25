# RTS Game - Auto-Opening Gates Feature

This directory contains the implementation of auto-opening gates for the Orca RTS game.

## Features

- **Automatic Gate Detection**: Gates automatically detect friendly units within a configurable radius
- **Smart Opening**: Gates open only for friendly units, staying closed for enemies
- **Timed Closing**: Gates automatically close after a configurable delay when no friendly units are nearby
- **Pathfinding Integration**: A* pathfinding algorithm considers gate states and unit ownership
- **Animation Support**: Gate open/close events can be hooked for visual animations

## Architecture

### Files

1. **`store/wallStore.ts`**: Central state management for walls, gates, and units
   - Manages gate states (open/closed)
   - Handles unit positions and ownership
   - Provides subscription system for state changes

2. **`buildings/WallSystem.tsx`**: React component for gate management
   - Detects friendly units near gates
   - Triggers gate opening/closing
   - Handles animations
   - Configurable detection radius and close delay

3. **`pathfinding/pathfinding.ts`**: A* pathfinding with gate support
   - Considers open/closed gates
   - Checks unit ownership for gate accessibility
   - Path smoothing and optimization
   - Line of sight calculations

## Usage

### Basic Setup

```typescript
import { WallSystem, WallSystemUtils } from './buildings/WallSystem';
import { wallStore } from './store/wallStore';

// In your game component
function Game() {
  return (
    <div>
      <WallSystem 
        detectionRadius={3.0}
        closeDelay={2000}
        updateInterval={100}
      />
      {/* Other game components */}
    </div>
  );
}
```

### Creating Gates and Units

```typescript
import { WallSystemUtils } from './buildings/WallSystem';

// Create a gate
const gateId = WallSystemUtils.createGate(
  { x: 10, y: 10 },
  'player1' // owner ID
);

// Create a unit
const unitId = WallSystemUtils.createUnit(
  { x: 5, y: 5 },
  'player1' // owner ID
);

// Move unit
WallSystemUtils.moveUnit(unitId, { x: 6, y: 6 });
```

### Pathfinding

```typescript
import { findPath } from './pathfinding/pathfinding';

const path = findPath(
  { x: 0, y: 0 },
  { x: 20, y: 20 },
  {
    unitOwnerId: 'player1',
    gridWidth: 50,
    gridHeight: 50,
    allowDiagonal: true
  }
);

if (path) {
  console.log('Path found:', path);
} else {
  console.log('No path available');
}
```

## Configuration

### WallSystem Props

- `detectionRadius` (default: 3.0): How close units need to be to trigger gate opening
- `closeDelay` (default: 2000ms): Delay before gate closes after unit passes
- `updateInterval` (default: 100ms): How often to check for units near gates

### Pathfinding Options

- `unitOwnerId`: Player/faction ID for ownership checks
- `gridWidth`: Width of the game grid
- `gridHeight`: Height of the game grid
- `allowDiagonal`: Enable diagonal movement (default: true)

## Implementation Details

### Gate Opening Logic

1. WallSystem checks all gates every update interval
2. For each gate, find nearby units within detection radius
3. If friendly units detected:
   - Open gate (if closed)
   - Schedule gate to close after delay
4. Gate closes automatically via timer when no units nearby

### Pathfinding Integration

- Friendly units can path through closed gates (they will auto-open)
- Enemy units cannot path through enemy gates
- Open gates are walkable by all units
- Diagonal movement requires both adjacent cardinal tiles to be walkable

## Events

The system dispatches custom events that can be used for animations:

```typescript
// Listen for gate open events
window.addEventListener('gate-open', (e) => {
  const { gateId } = e.detail;
  // Play open animation
});

// Listen for gate close events
window.addEventListener('gate-close', (e) => {
  const { gateId } = e.detail;
  // Play close animation
});
```

## Testing

To test the implementation:

1. Create gates and units with the same owner ID
2. Move units toward gates
3. Observe gates opening when units approach
4. Observe gates closing after units pass
5. Test with enemy units (different owner ID) to verify gates stay closed

## Future Enhancements

- Alliance system for shared gate access
- Gate health and destructibility
- Different gate types (portcullis, drawbridge, etc.)
- Batch gate operations for performance
- Network synchronization for multiplayer
