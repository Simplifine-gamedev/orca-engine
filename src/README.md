# Orca RTS - Mob Lair Spawning System

This directory contains the implementation of the mob lair spawning mechanics for Orca RTS.

## Features

- **Multiple Lair Types**: Goblin camps, ogre caves, undead crypts, and dragon nests
- **Periodic Mob Spawning**: Each lair spawns mobs at configurable intervals
- **Health System**: Lairs can be damaged and destroyed
- **Loot Drops**: Destroying lairs drops randomized loot based on loot tables
- **Multiplayer Support**: Real-time synchronization via WebSocket
- **State Management**: Zustand-based store for efficient state updates

## Architecture

### Client-Side

```
src/
├── objects/
│   └── MobLair.tsx        # Main lair component with spawn logic
├── store/
│   └── mobStore.ts        # Zustand store for lairs and mobs
└── components/
    └── GameWorld.tsx      # Game world container with networking
```

### Server-Side

```
server/
├── GameServer.js          # Express + Socket.IO game server
└── package.json           # Server dependencies
```

## Lair Types

### Goblin Camp ⛺
- **Health**: 500
- **Spawn Interval**: 30 seconds
- **Max Mobs**: 8
- **Mob Type**: Goblin
- **Loot**: Gold, goblin dagger, leather scraps

### Ogre Cave 🕳️
- **Health**: 1200
- **Spawn Interval**: 60 seconds
- **Max Mobs**: 4
- **Mob Type**: Ogre
- **Loot**: Gold, ogre club, thick hide

### Undead Crypt ⚰️
- **Health**: 800
- **Spawn Interval**: 25 seconds
- **Max Mobs**: 12
- **Mob Type**: Skeleton
- **Loot**: Gold, bone sword, soul essence

### Dragon Nest 🏔️
- **Health**: 3000
- **Spawn Interval**: 120 seconds
- **Max Mobs**: 2
- **Mob Type**: Dragon
- **Loot**: Gold, dragon scale, legendary weapon

## Usage

### Starting the Server

```bash
cd server
npm install
npm start
```

The server will start on port 3001 by default.

### Creating a Lair (Client)

```typescript
import { useMobStore } from './store/mobStore';
import { LAIR_TYPES } from './objects/MobLair';

const { addLair } = useMobStore();

// Create a goblin camp
addLair({
  id: 'lair_1',
  ...LAIR_TYPES.goblin_camp,
  position: { x: 400, y: 300 },
  health: 500,
  isDestroyed: false,
});
```

### Using the GameWorld Component

```tsx
import React from 'react';
import { GameWorld } from './components/GameWorld';

function App() {
  return (
    <div className="App">
      <GameWorld serverUrl="http://localhost:3001" />
    </div>
  );
}

export default App;
```

## API Reference

### MobStore

#### Lair Operations
- `addLair(lair: MobLairConfig)` - Add a new lair
- `removeLair(lairId: string)` - Remove a lair
- `updateLair(lairId: string, updates: Partial<MobLairConfig>)` - Update lair properties
- `destroyLair(lairId: string)` - Mark a lair as destroyed
- `getLair(lairId: string)` - Get a specific lair
- `getAllLairs()` - Get all lairs

#### Mob Operations
- `spawnMob(mob: Mob)` - Spawn a new mob
- `removeMob(mobId: string)` - Remove a mob
- `updateMob(mobId: string, updates: Partial<Mob>)` - Update mob properties
- `killMob(mobId: string)` - Kill a mob
- `getMob(mobId: string)` - Get a specific mob
- `getAllMobs()` - Get all mobs
- `getMobsByLair(lairId: string)` - Get mobs from a specific lair
- `getAliveMobs()` - Get all alive mobs

### Server Events

#### Incoming (Client → Server)
- `lair:damage` - Damage a lair
- `mob:damage` - Damage a mob
- `mob:move` - Update mob position

#### Outgoing (Server → Client)
- `game:state` - Initial game state on connection
- `lair:created` - New lair created
- `lair:damaged` - Lair took damage
- `lair:destroyed` - Lair was destroyed with loot
- `mob:spawned` - New mob spawned
- `mob:damaged` - Mob took damage
- `mob:killed` - Mob was killed
- `mob:moved` - Mob moved to new position

## REST API Endpoints

### GET /api/game-state
Get current game state including all lairs and mobs.

### POST /api/lairs
Create a new lair.

**Request Body:**
```json
{
  "type": "goblin_camp",
  "position": { "x": 400, "y": 300 }
}
```

### DELETE /api/lairs/:lairId
Remove a lair and all its mobs.

## Configuration

### Environment Variables

**Server:**
- `PORT` - Server port (default: 3001)

**Client:**
- `REACT_APP_SERVER_URL` - Game server URL (default: http://localhost:3001)

## Customization

### Adding New Lair Types

1. Add lair configuration to `LAIR_TYPES` in `src/objects/MobLair.tsx`
2. Add server-side config in `server/GameServer.js`
3. Add sprite/icon in the lair sprite function

Example:
```typescript
vampire_castle: {
  type: 'vampire_castle',
  maxHealth: 2000,
  spawnInterval: 45000,
  mobType: 'vampire',
  maxMobs: 6,
  lootTable: [
    { item: 'blood_ruby', quantity: 3, dropChance: 0.8 },
    { item: 'vampire_cloak', quantity: 1, dropChance: 0.2 },
  ],
}
```

### Adjusting Spawn Rates

Modify the `spawnInterval` property in the lair configuration (in milliseconds):
- 10000 = 10 seconds
- 30000 = 30 seconds
- 60000 = 1 minute

## Testing

### Manual Testing
1. Start the server
2. Open multiple browser tabs
3. Create lairs using the UI buttons
4. Click on lairs to damage them
5. Click on mobs to damage them
6. Verify synchronization across tabs

### Automated Tests
```bash
cd server
npm test
```

## Performance Considerations

- Lairs stop spawning when destroyed
- Dead mobs are automatically cleaned up after 2 seconds
- Server runs a game loop every 5 seconds for mob AI
- Destroyed lairs are cleaned up after 60 seconds

## Future Enhancements

- [ ] Mob AI pathfinding
- [ ] Player targeting system
- [ ] Lair repair/reconstruction
- [ ] Lair upgrade system
- [ ] Wave-based spawning
- [ ] Boss lairs with unique mechanics
- [ ] Territory control mechanics
- [ ] Resource gathering from defeated mobs

## License

MIT
