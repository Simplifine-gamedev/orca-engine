# Rally Point to Resource Feature

## Overview

This feature allows players to set rally points directly on resources (like gold mines) so that newly spawned workers automatically start gathering from those resources.

## Implementation

### Files Modified/Created

1. **src/store/gameStore.ts** - Core game state management
   - Handles rally point detection on resources
   - Automatically assigns spawned workers to gather if rally point is on a resource
   - Provides resource position detection within a radius

2. **src/buildings/Building.tsx** - Building component with visual indicators
   - Displays rally point markers
   - Shows special visual indicator when rally point is on a resource (gold pickaxe icon)
   - Provides UI controls for setting rally points and spawning workers

3. **server/GameServer.js** - Server-side multiplayer support
   - Synchronizes rally points across all players
   - Handles resource detection server-side
   - Broadcasts unit spawning with automatic resource assignment

## Features

### 1. Resource Detection
When setting a rally point, the system automatically detects if it's placed on a resource (within 50 pixels radius).

### 2. Visual Indicators
- **Regular rally point**: Green flag marker with dashed line
- **Resource rally point**: Gold glowing circle with pickaxe icon
- **Resource type label**: Shows "Gathering [resource type]" below the marker

### 3. Automatic Worker Assignment
When a worker is spawned from a building with a rally point on a resource:
- Worker is immediately assigned to gather mode
- Worker moves to the resource location
- Worker's `isGathering` flag is set to true
- Worker's `targetResourceId` is set to the resource

### 4. Multiplayer Support
Server synchronizes:
- Rally point updates
- Resource locations
- Unit spawning and assignment
- Building states

## Usage

### Setting a Rally Point

```typescript
import { gameStore } from './store/gameStore';

// Set rally point on a position
gameStore.setRallyPoint(buildingId, { x: 300, y: 200 });

// If position is on a resource, it's automatically detected
```

### Spawning Workers

```typescript
// Spawn a worker - automatically assigned if rally point is on resource
const unit = gameStore.spawnUnit(buildingId, 'worker');

if (unit.isGathering) {
  console.log(`Worker gathering from resource: ${unit.targetResourceId}`);
}
```

### React Component Usage

```tsx
import { Building } from './buildings/Building';

<Building
  building={building}
  onSelect={() => gameStore.selectBuilding(building.id)}
  onSetRallyPoint={(pos) => gameStore.setRallyPoint(building.id, pos)}
/>
```

## Server Setup

Start the game server:

```bash
cd server
npm install
node GameServer.js
```

The server will:
- Listen on port 3001 (configurable via PORT env var)
- Initialize test resources (gold mine, wood pile)
- Handle real-time multiplayer synchronization

## Technical Details

### Resource Detection Algorithm

```typescript
private findResourceAtPosition(position: { x: number; y: number }): Resource | undefined {
  const RESOURCE_RADIUS = 50; // Detection radius

  for (const resource of this.state.resources.values()) {
    const distance = Math.sqrt(
      Math.pow(resource.position.x - position.x, 2) +
      Math.pow(resource.position.y - position.y, 2)
    );

    if (distance <= RESOURCE_RADIUS) {
      return resource;
    }
  }

  return undefined;
}
```

### Rally Point Data Structure

```typescript
interface RallyPoint {
  position: { x: number; y: number };
  targetResourceId?: string;      // Set if on resource
  targetResource?: Resource;       // Full resource reference
}
```

## Testing

1. Create a building
2. Create a resource nearby
3. Set rally point on the resource
4. Spawn a worker
5. Verify worker is automatically assigned to gather

## Future Enhancements

- Support for multiple resource types (gold, wood, stone)
- Rally point queue for multiple resources
- Visual path showing worker movement to resource
- Resource depletion and worker reassignment
- Rally point persistence across game sessions
