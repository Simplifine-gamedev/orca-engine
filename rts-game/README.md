# Orca RTS - Mob Lair Spawning System

A real-time strategy game featuring mob lairs that continuously spawn mobs for farming.

## Features

### Mob Lairs
- **5 Different Lair Types**:
  - **Goblin Camp**: Spawns goblin warriors and archers (Level 1-3)
  - **Ogre Cave**: Spawns ogres and cave trolls (Level 5-8)
  - **Wolf Den**: Spawns wolves and dire wolves (Level 2-6)
  - **Bandit Hideout**: Spawns bandits and rogues (Level 3-6)
  - **Undead Crypt**: Spawns skeletons, zombies, and ghouls (Level 2-7)

### Core Mechanics
- **Periodic Spawning**: Each lair spawns mobs at configurable intervals
- **Max Mob Limits**: Lairs have maximum mob counts to prevent overwhelming
- **Destructible Lairs**: Destroy lairs to stop spawning and get loot
- **Spawn Radius**: Mobs spawn within a radius around the lair
- **Health System**: Lairs have health and can be damaged
- **Loot Tables**: Destroying lairs drops rewards based on lair type

## Project Structure

```
rts-game/
├── src/
│   ├── objects/
│   │   └── MobLair.tsx          # Lair component with types and configs
│   └── store/
│       └── mobStore.ts          # State management and spawning logic
├── server/
│   └── GameServer.js            # Server-side game logic and Socket.IO
├── package.json
└── README.md
```

## Installation

```bash
cd rts-game
npm install
```

## Usage

### Starting the Server

```bash
npm start
```

The server will start on port 3001 (configurable via PORT environment variable).

### Development Mode

```bash
npm run dev
```

Uses nodemon for auto-reloading during development.

## API Reference

### Server Endpoints

#### GET /health
Returns server health status and statistics.

Response:
```json
{
  "status": "ok",
  "uptime": 12345,
  "lairs": 3,
  "mobs": 12,
  "players": 2
}
```

#### GET /state
Returns current game state.

Response:
```json
{
  "lairs": [...],
  "mobs": [...],
  "players": 2
}
```

### Socket.IO Events

#### Client → Server

- `lair:create` - Create a new lair
  ```javascript
  socket.emit('lair:create', {
    type: 'goblin_camp',
    position: { x: 100, y: 200 }
  });
  ```

- `lair:damage` - Damage a lair
  ```javascript
  socket.emit('lair:damage', {
    lairId: 'lair_123',
    damage: 50
  });
  ```

- `lair:destroy` - Destroy a lair
  ```javascript
  socket.emit('lair:destroy', {
    lairId: 'lair_123'
  });
  ```

- `mob:kill` - Kill a mob
  ```javascript
  socket.emit('mob:kill', {
    mobId: 'mob_456'
  });
  ```

- `mob:damage` - Damage a mob
  ```javascript
  socket.emit('mob:damage', {
    mobId: 'mob_456',
    damage: 30
  });
  ```

- `game:reset` - Reset game state

#### Server → Client

- `game:state` - Initial game state sent on connection
- `lair:created` - New lair created
- `lair:damaged` - Lair took damage
- `lair:destroyed` - Lair destroyed with loot
- `mob:spawned` - New mob spawned
- `mob:killed` - Mob was killed
- `mob:damaged` - Mob took damage
- `game:reset` - Game was reset

## Client-Side Usage

### Using the Mob Store

```typescript
import { useMobStore } from './src/store/mobStore';

function GameComponent() {
  const { createLair, startSpawning, getAllLairs } = useMobStore();

  useEffect(() => {
    // Create some lairs
    createLair(LairType.GOBLIN_CAMP, { x: 100, y: 100 });
    createLair(LairType.WOLF_DEN, { x: 300, y: 200 });

    // Start the spawning system
    startSpawning();

    return () => {
      stopSpawning();
    };
  }, []);

  const lairs = getAllLairs();

  return (
    <div>
      {lairs.map(lair => (
        <MobLair
          key={lair.id}
          lair={lair}
          onDestroy={(id) => destroyLair(id)}
        />
      ))}
    </div>
  );
}
```

### Rendering Lairs

```typescript
import MobLair from './src/objects/MobLair';

<MobLair
  lair={lairConfig}
  onDestroy={(id) => console.log(`Lair ${id} destroyed`)}
  onTakeDamage={(id, damage) => console.log(`Lair ${id} took ${damage} damage`)}
  onClick={(id) => console.log(`Lair ${id} clicked`)}
/>
```

## Lair Configuration

Each lair type has unique properties defined in `LAIR_CONFIGS`:

```typescript
{
  displayName: string;      // UI display name
  maxHealth: number;        // Maximum health points
  spawnInterval: number;    // Milliseconds between spawns
  maxMobs: number;          // Max concurrent mobs
  spawnRadius: number;      // Spawn area radius
  mobSpawns: [{             // Mob types to spawn
    mobType: string;
    level: { min, max };
    count: number;
  }];
  lootTable: [{             // Loot drops on destruction
    itemId: string;
    chance: number;         // 0-1 probability
    quantity: { min, max };
  }];
  appearance: {
    color: string;          // Visual color
    size: number;           // Display size
  };
}
```

## Game Balance

### Spawn Rates
- **Goblin Camp**: 30s interval, up to 5 mobs
- **Ogre Cave**: 60s interval, up to 3 mobs
- **Wolf Den**: 20s interval, up to 8 mobs
- **Bandit Hideout**: 40s interval, up to 6 mobs
- **Undead Crypt**: 45s interval, up to 10 mobs

### Difficulty Progression
- Lower-tier lairs (Goblin, Wolf): Easier mobs, faster spawns
- Mid-tier lairs (Bandit): Moderate difficulty
- High-tier lairs (Ogre, Undead): Tougher mobs, better rewards

## Development

### Adding New Lair Types

1. Add new `LairType` enum value in `MobLair.tsx`
2. Define configuration in `LAIR_CONFIGS`
3. Add mob type definitions
4. Update server lair configs in `GameServer.js`

### Adding New Mob Types

1. Add mob type to lair's `mobSpawns` configuration
2. Define base health in `calculateMobHealth` function
3. Add mob visuals/rendering logic

## Testing

The implementation includes:
- Automatic spawn timers
- Mob count limits per lair
- Health and damage systems
- Loot generation on lair destruction
- Real-time multiplayer synchronization

## Architecture

### Client-Side (Zustand Store)
- Manages game state (lairs, mobs)
- Handles spawning logic
- Provides reactive state updates

### Server-Side (Socket.IO)
- Authoritative game server
- Handles multiplayer synchronization
- Manages spawn timers server-side
- Broadcasts state changes to all clients

## License

MIT

## Credits

Developed for Orca RTS game engine.
