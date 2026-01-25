# Orca RTS - Auto-Opening Gates Feature

This project implements an auto-opening gate system for the Orca RTS game, where gates automatically open when friendly units approach and close after they pass.

## Features Implemented

### 1. Friendly Unit Detection Near Gates
- Gates detect friendly units within a configurable radius (default: 3 tiles)
- Detection runs automatically every 100ms
- Only friendly units trigger gate opening

### 2. Automatic Gate Opening Animation
- Smooth animation when gates open/close
- Visual feedback with color changes:
  - **Closed**: Orange (#ed8936)
  - **Open**: Green (#48bb78)
- Scale animation (0.8x when open, 1x when closed)
- Opacity change for better visual distinction

### 3. Pathfinding Through Open Gates
- A* pathfinding algorithm with gate awareness
- Friendly units can path through open gates
- Enemy units cannot path through gates (even if open)
- Dynamic pathfinding updates based on gate state

### 4. Auto-Close with Delay
- Gates automatically close after units pass
- Configurable delay (default: 2 seconds)
- Delay resets if more units approach
- Smooth transition animations

### 5. Enemy Unit Blocking
- Enemy units cannot pass through gates
- Pathfinding treats gates as impassable for enemies
- No gate opening triggered by enemy units

## File Structure

```
src/
├── types/
│   └── index.ts              # TypeScript type definitions
├── store/
│   └── wallStore.ts          # State management for walls/gates
├── pathfinding/
│   └── pathfinding.ts        # A* pathfinding algorithm
├── buildings/
│   └── WallSystem.tsx        # React component for rendering
├── utils/
│   └── unitManager.ts        # Unit movement management
├── demo/
│   └── GameDemo.tsx          # Demo application
├── main.tsx                  # Application entry point
├── index.html                # HTML template
├── package.json              # Dependencies
├── tsconfig.json             # TypeScript config
└── vite.config.ts            # Vite build config
```

## Architecture

### State Management (`wallStore.ts`)
- Centralized store for walls, gates, and units
- Subscribe/notify pattern for reactive updates
- Automatic gate checking system
- Methods for adding/removing walls and units
- Position blocking logic for pathfinding

### Pathfinding (`pathfinding.ts`)
- A* algorithm implementation
- 4-directional movement (up, down, left, right)
- Manhattan distance heuristic
- Team-aware pathfinding (friendly vs enemy)
- Dynamic obstacle checking based on gate states

### Rendering (`WallSystem.tsx`)
- React component for visual representation
- Real-time updates via store subscription
- Smooth animations using CSS transitions
- Debug overlay showing system status
- Manual gate control for testing

### Unit Management (`unitManager.ts`)
- Create and manage units
- Path-based movement system
- Automatic movement updates
- Integration with wallStore for real-time updates

## Usage

### Installation

```bash
cd src
npm install
```

### Development

```bash
npm run dev
```

This will start a development server at `http://localhost:3000`.

### Building

```bash
npm run build
```

### Type Checking

```bash
npm run type-check
```

## API Reference

### WallStore

```typescript
// Add a wall or gate
wallStore.addWall(wall: Wall): void

// Remove a wall or gate
wallStore.removeWall(id: string): void

// Update unit position
wallStore.updateUnit(unit: Unit): void

// Open/close gates
wallStore.openGate(gateId: string): void
wallStore.closeGate(gateId: string): void

// Check if position is blocked
wallStore.isPositionBlocked(position: Position, team: 'friendly' | 'enemy'): boolean

// Start/stop automatic gate checking
wallStore.startGateChecking(intervalMs?: number): void
wallStore.stopGateChecking(): void
```

### Pathfinder

```typescript
// Find path between two points
pathfinder.findPath(start: Position, goal: Position, team: 'friendly' | 'enemy'): Position[] | null

// Check if path exists
pathfinder.hasPath(start: Position, goal: Position, team: 'friendly' | 'enemy'): boolean

// Get next position for unit movement
pathfinder.getNextPosition(currentPos: Position, goal: Position, team: 'friendly' | 'enemy'): Position | null

// Update grid size
pathfinder.setGridSize(width: number, height: number): void
```

### UnitManager

```typescript
// Create a unit
unitManager.createUnit(id: string, position: Position, team: 'friendly' | 'enemy'): Unit

// Move unit to target
unitManager.moveUnit(unitId: string, target: Position): boolean

// Start/stop automatic movement
unitManager.startMovementUpdates(intervalMs?: number): void
unitManager.stopMovementUpdates(): void

// Get units
unitManager.getUnits(): Unit[]
unitManager.getUnit(unitId: string): Unit | undefined
```

## Configuration

### Gate Properties

Gates can be configured with the following properties:

```typescript
interface Gate {
  id: string;
  position: Position;
  type: 'gate';
  isOpen: boolean;
  team: 'friendly' | 'neutral';
  closeDelay: number;        // Delay before auto-close (ms)
  detectionRadius: number;   // Radius to detect units (tiles)
}
```

Default values:
- `closeDelay`: 2000ms (2 seconds)
- `detectionRadius`: 3 tiles

### System Update Intervals

```typescript
// Gate checking interval (default: 100ms)
wallStore.startGateChecking(100);

// Unit movement interval (default: 200ms)
unitManager.startMovementUpdates(300);
```

## Testing

The demo application (`GameDemo.tsx`) includes:
- Pre-configured wall with gate in the middle
- 3 friendly units (blue with shields)
- 1 enemy unit (red with sword)
- Buttons to test movement through gates

### Manual Testing

1. Click "Move Units Through Gate" to send friendly units through the gate
2. Observe the gate automatically opening when units approach
3. Watch the gate close after units pass (2-second delay)
4. Try moving enemy units - they cannot pass through gates

## Implementation Details

### Auto-Opening Logic

The gate opening logic is implemented in `wallStore.ts`:

1. Every 100ms, check all gates for nearby friendly units
2. If friendly units are within detection radius and gate is closed → open gate
3. If gate is open and no units nearby → start close timer
4. If units remain nearby → reset close timer
5. After delay expires and no units nearby → close gate

### Pathfinding Integration

The pathfinding system in `pathfinding.ts`:

1. Uses A* algorithm for optimal path finding
2. Checks `wallStore.isPositionBlocked()` for each tile
3. Gates are treated as:
   - **Passable** for friendly units when open
   - **Blocked** for friendly units when closed
   - **Always blocked** for enemy units
4. Paths are recalculated when gate states change

## Performance Considerations

- Gate checking runs at 100ms intervals (10 FPS)
- Only checks gates that exist (not all tiles)
- Distance calculations use simple Euclidean distance
- Pathfinding uses Manhattan distance heuristic
- Store updates use efficient Map structures
- Animations use CSS transitions (GPU-accelerated)

## Future Enhancements

Potential improvements:
- Multiple gate types (small, large, reinforced)
- Gate health/damage system
- Sound effects for gate opening/closing
- Particle effects for animations
- Multiplayer support with team-based gates
- Gate upgrade system
- Hotkey controls for manual gate operation

## License

Part of the Orca RTS Game project.
