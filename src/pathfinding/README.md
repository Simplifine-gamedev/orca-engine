# Orca RTS Pathfinding System

Advanced pathfinding implementation for the Orca RTS game, addressing all known pathfinding issues.

## Features

### ✅ Solved Issues

1. **Better Obstacle Avoidance** - Units now avoid obstacles more intelligently using weighted cost fields
2. **Group Pathfinding Optimization** - Multiple units can pathfind efficiently with spatial separation
3. **Dynamic Obstacle Handling** - Real-time obstacle updates with smart cache invalidation
4. **Path Smoothing** - Line-of-sight smoothing removes unnecessary waypoints
5. **Construction State Support** - Integrates construction states into pathfinding cache hash (commit 5434cdc)

### Core Components

#### 1. `Pathfinder` (pathfinding.ts)
Main synchronous pathfinding class using improved A* algorithm.

**Key Features:**
- Octile distance heuristic for better diagonal movement
- Obstacle avoidance cost field
- Construction state tracking
- Path caching with obstacle hash
- Line-of-sight path smoothing
- Group pathfinding with unit separation

**Usage:**
```typescript
import { createPathfinder } from './pathfinding';

const pathfinder = createPathfinder(1.0); // 1.0 = grid size

// Add obstacles
pathfinder.addObstacle('building1', {
  x: 10,
  y: 10,
  radius: 3,
  constructionState: 'complete'
});

// Find path
const path = pathfinder.findPath(
  { x: 0, y: 0 },
  { x: 20, y: 20 },
  {
    unitRadius: 0.5,
    allowDiagonal: true,
    smoothPath: true,
    avoidanceWeight: 0.5
  }
);
```

#### 2. `AsyncPathfinder` (pathfindingAsync.ts)
Non-blocking pathfinding with request queuing and time-slicing.

**Key Features:**
- Promise-based API
- Priority queuing
- Time-sliced execution
- Batch processing
- Prevents frame drops in RTS games

**Usage:**
```typescript
import { createPathfinder, createAsyncPathfinder } from './pathfinding';

const pathfinder = createPathfinder(1.0);
const asyncPathfinder = createAsyncPathfinder(pathfinder);

// Request path asynchronously
const path = await asyncPathfinder.requestPath(
  'unit1',
  { x: 0, y: 0 },
  { x: 20, y: 20 },
  { unitRadius: 0.5 },
  priority: 1 // Higher priority = processed first
);
```

#### 3. `ProgressivePathfinder` (pathfindingAsync.ts)
For very large unit groups with progress tracking.

**Usage:**
```typescript
import { createProgressivePathfinder } from './pathfinding';

const progressivePathfinder = createProgressivePathfinder(asyncPathfinder);

const results = await progressivePathfinder.findGroupPathsProgressive(
  units,
  { unitRadius: 0.5 },
  (progress, completed, total) => {
    console.log(`Pathfinding: ${completed}/${total} (${(progress * 100).toFixed(1)}%)`);
  }
);
```

#### 4. `FlowFieldPathfinder` (pathfindingAsync.ts)
Efficient pathfinding for large groups moving to same destination.

**Usage:**
```typescript
import { createFlowFieldPathfinder } from './pathfinding';

const flowFieldPathfinder = createFlowFieldPathfinder(pathfinder);

// Generate flow field
const flowField = flowFieldPathfinder.generateFlowField(
  { x: 100, y: 100 }, // Goal
  { minX: 0, maxX: 200, minY: 0, maxY: 200 }, // Bounds
  1.0, // Grid size
  0.5  // Unit radius
);

// Get direction for a unit
const direction = flowFieldPathfinder.getFlowDirection(
  flowField,
  { x: 50, y: 50 },
  1.0
);
```

## Pathfinding Options

```typescript
interface PathfindingOptions {
  unitRadius?: number;         // Unit collision radius (default: 0.5)
  allowDiagonal?: boolean;     // Allow diagonal movement (default: true)
  heuristicWeight?: number;    // A* heuristic weight (default: 1.0)
  smoothPath?: boolean;        // Apply path smoothing (default: true)
  avoidanceWeight?: number;    // Obstacle avoidance strength (default: 0.5)
  maxIterations?: number;      // Max A* iterations (default: 10000)
}
```

## Dynamic Obstacles

The system supports dynamic obstacles that can move during gameplay:

```typescript
// Add dynamic obstacle
pathfinder.addObstacle('unit1', {
  x: 10,
  y: 10,
  radius: 0.5,
  isDynamic: true,
  constructionState: 'building'
});

// Update obstacle position
pathfinder.updateObstacle('unit1', { x: 11, y: 10 }, 'complete');

// Remove obstacle
pathfinder.removeObstacle('unit1');
```

## Group Pathfinding

Efficient pathfinding for multiple units with automatic separation:

```typescript
const units = [
  { id: 'unit1', start: { x: 0, y: 0 }, goal: { x: 20, y: 20 } },
  { id: 'unit2', start: { x: 1, y: 0 }, goal: { x: 21, y: 20 } },
  { id: 'unit3', start: { x: 2, y: 0 }, goal: { x: 22, y: 20 } }
];

const paths = pathfinder.findGroupPaths(units, {
  unitRadius: 0.5,
  smoothPath: true
});

// Or async
const asyncPaths = await asyncPathfinder.requestGroupPaths(units, {
  unitRadius: 0.5
});
```

## Performance Tips

1. **Use Async Pathfinding** - Prevents frame drops during heavy pathfinding
2. **Enable Path Caching** - Automatically enabled, reuses paths when obstacles haven't changed
3. **Use Flow Fields** - For 50+ units moving to same goal
4. **Tune Grid Size** - Larger grid = faster pathfinding but less precise
5. **Adjust Max Processing Time** - Balance between responsiveness and throughput

```typescript
// Configure async pathfinder performance
asyncPathfinder.setProcessingParams(
  20,  // Max batch size
  10   // Max processing time per frame (ms)
);
```

## Integration with Construction System

The pathfinding system integrates with the construction system (commit 5434cdc):

```typescript
// Obstacles automatically include construction state in cache hash
pathfinder.addObstacle('building_under_construction', {
  x: 50,
  y: 50,
  radius: 5,
  constructionState: 'foundation'
});

// Update construction state
pathfinder.updateObstacle('building_under_construction', 
  { x: 50, y: 50 }, 
  'walls'
);

// Cache is automatically invalidated when construction state changes
```

## Debugging

Get statistics about the pathfinding system:

```typescript
const stats = pathfinder.getStats();
console.log('Obstacles:', stats.obstacleCount);
console.log('Cached paths:', stats.cacheSize);
console.log('Dynamic obstacles:', stats.dynamicObstacleCount);

const asyncStats = asyncPathfinder.getStats();
console.log('Queue size:', asyncStats.queueSize);
console.log('Processing:', asyncStats.processing);
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Game Layer                        │
│  (Units, Buildings, Construction, Game Logic)       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Async Pathfinding Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │AsyncPathfinder│  │Progressive   │  │Flow Field │ │
│  │              │  │Pathfinder    │  │Pathfinder │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Core Pathfinding Layer                  │
│  ┌──────────────────────────────────────────────┐  │
│  │            Pathfinder (A*)                   │  │
│  │  - Obstacle Management                       │  │
│  │  - Path Caching                              │  │
│  │  - Path Smoothing                            │  │
│  │  - Group Pathfinding                         │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Future Improvements

- [ ] Hierarchical pathfinding for very large maps
- [ ] Jump point search optimization
- [ ] Pathfinding debug visualization
- [ ] Steering behaviors for unit movement
- [ ] Terrain cost modifiers (slow/fast terrain)

## License

Part of the Orca RTS Engine - MIT License
