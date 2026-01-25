# Orca RTS - Rally Point to Resource Gathering

## Overview

This implementation adds the ability to set rally points directly on resources (like gold mines) so that spawned workers automatically start gathering from those resources.

## Features Implemented

### 1. Resource Detection at Rally Points
- When setting a rally point, the system detects if there's a resource within a 50-unit radius
- Resources are identified by their position and type (gold, wood, stone)

### 2. Automatic Worker Assignment
- Workers spawned from buildings with resource rally points are automatically assigned to gather
- The worker's `isGathering` flag is set to `true`
- The worker's `targetResource` is set to the detected resource ID

### 3. Visual Indicators
- **Yellow pulsing dot**: Indicates a rally point is set on a resource
- **White pulsing dot**: Indicates a regular rally point (no resource)
- **Dashed line**: Shows the connection between the building and its rally point
- **Resource label**: Displays the resource type (e.g., "GOLD") at resource rally points
- **Worker status**: Workers show a gathering indicator when assigned to a resource

## File Structure

```
rts-game/
├── src/
│   ├── store/
│   │   └── gameStore.ts           # Game state management with rally point logic
│   ├── buildings/
│   │   └── Building.tsx           # Building component with rally point UI
│   ├── components/
│   │   ├── Game.tsx               # Main game component
│   │   ├── Resource.tsx           # Resource visualization component
│   │   └── Unit.tsx               # Unit/worker component
│   └── types/
│       └── index.ts               # TypeScript type definitions
├── server/
│   └── GameServer.js              # Multiplayer game server with rally point sync
├── app/
│   ├── layout.tsx                 # Next.js app layout
│   ├── page.tsx                   # Main page
│   └── globals.css                # Global styles
└── package.json                   # Dependencies and scripts
```

## Implementation Details

### Rally Point Detection (gameStore.ts)

```typescript
detectResourceAtPosition: (position: Position): Resource | null => {
  const { resources } = get();
  
  for (const [_, resource] of resources) {
    const distance = Math.sqrt(
      Math.pow(resource.position.x - position.x, 2) +
      Math.pow(resource.position.y - position.y, 2)
    );
    
    if (distance <= RESOURCE_DETECTION_RADIUS) {
      return resource;
    }
  }
  
  return null;
}
```

### Rally Point Setting

When a rally point is set, the system:
1. Calculates the position where the user clicked
2. Checks for nearby resources using `detectResourceAtPosition()`
3. Creates a `RallyPoint` object with:
   - `position`: The click coordinates
   - `targetResource`: The detected resource (if any)
   - `isResourceRallyPoint`: Boolean flag indicating if it's a resource rally point

### Worker Spawning

When a worker is spawned:
1. It's created at the building's position
2. If the building has a rally point:
   - The worker moves to the rally point position
   - If it's a resource rally point, the worker is automatically assigned to gather
   - The worker's `isGathering` flag is set to `true`
   - The `targetResource` is set to the resource ID

### Visual Feedback

The Building component renders:
- A dashed line from the building to the rally point
- A colored indicator at the rally point (yellow for resource, white for regular)
- A label showing the resource type at resource rally points
- An info panel showing rally point coordinates and resource info

## Usage

### Setting a Rally Point

1. Click the "Set Rally Point" button on a building
2. Click anywhere on the map to set the rally point
3. If you click near a resource, it will automatically become a resource rally point

### Spawning Workers

1. Click the "Spawn Worker" button on a building
2. The worker will appear at the rally point (or building if no rally point is set)
3. If the rally point is on a resource, the worker will automatically start gathering

## Running the Project

### Development Mode

```bash
# Install dependencies
npm install

# Run the development server
npm run dev

# In another terminal, run the game server (optional, for multiplayer)
npm run server
```

Open [http://localhost:3000](http://localhost:3000) to see the game.

### Production Build

```bash
# Build the application
npm run build

# Start the production server
npm start
```

## Technical Stack

- **Next.js 14**: React framework for the frontend
- **TypeScript**: Type-safe development
- **Zustand**: Lightweight state management
- **Tailwind CSS**: Utility-first CSS framework
- **WebSockets**: Real-time multiplayer communication (server)

## Game Server

The `GameServer.js` file implements a WebSocket server that:
- Synchronizes game state across multiple clients
- Handles rally point setting and unit spawning on the server side
- Detects resources at rally points server-side for authoritative game logic
- Broadcasts state changes to all connected clients

## Future Enhancements

- Add pathfinding for workers to navigate to resources
- Implement actual resource gathering mechanics (decrease resource amount, increase player resources)
- Add multiple building types with different rally point behaviors
- Implement unit selection and manual resource assignment
- Add fog of war and resource discovery
- Optimize performance for large numbers of units
- Add animations for unit movement and gathering

## Linear Issue Resolution

This implementation fully resolves Linear issue **ORC-120**:

- ✅ Detect when rally point is set on a resource
- ✅ Spawned workers automatically assigned to gather from that resource
- ✅ Visual indicator showing rally point is on a resource

## License

MIT
