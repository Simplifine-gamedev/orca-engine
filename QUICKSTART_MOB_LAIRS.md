# Quick Start Guide - Mob Lair System

Get up and running with the mob lair spawning mechanics in 5 minutes!

## Installation

### 1. Install Server Dependencies

```bash
cd server
npm install
```

### 2. Install Client Dependencies

```bash
cd ../src
npm install
```

## Running the System

### Start the Game Server

```bash
cd server
npm start
```

The server will start on `http://localhost:3001`

### Use in Your React App

```tsx
import React from 'react';
import { GameWorld } from './src/components/GameWorld';
import './src/styles/game.css';

function App() {
  return (
    <div className="App">
      <GameWorld serverUrl="http://localhost:3001" />
    </div>
  );
}

export default App;
```

## Basic Usage Examples

### Example 1: Simple Game with UI Controls

Use the `GameWorld` component which includes a UI for creating lairs:

```tsx
import { GameWorld } from './src/components/GameWorld';

<GameWorld serverUrl="http://localhost:3001" />
```

Click the buttons in the top-left to create different lair types!

### Example 2: Programmatic Lair Creation

Create lairs directly in code:

```tsx
import { useMobStore } from './src/store/mobStore';
import { LAIR_TYPES } from './src/objects/MobLair';

function MyGame() {
  const { addLair } = useMobStore();

  const createGoblinCamp = () => {
    addLair({
      id: `goblin_${Date.now()}`,
      ...LAIR_TYPES.goblin_camp,
      position: { x: 400, y: 300 },
      health: 500,
      isDestroyed: false,
    });
  };

  return <button onClick={createGoblinCamp}>Spawn Goblin Camp</button>;
}
```

### Example 3: Server-Side Lair Creation (REST API)

```bash
# Create a goblin camp at position (400, 300)
curl -X POST http://localhost:3001/api/lairs \
  -H "Content-Type: application/json" \
  -d '{"type": "goblin_camp", "position": {"x": 400, "y": 300}}'

# Get current game state
curl http://localhost:3001/api/game-state

# Delete a lair
curl -X DELETE http://localhost:3001/api/lairs/LAIR_ID
```

## How It Works

1. **Lairs spawn mobs** at regular intervals
2. **Each lair type** has different properties:
   - Goblin Camp: Fast spawning, weak mobs
   - Ogre Cave: Slow spawning, strong mobs
   - Undead Crypt: Very fast spawning, many weak mobs
   - Dragon Nest: Very slow spawning, powerful mobs

3. **Damaging lairs**: Click on a lair to damage it
4. **Destroying lairs**: When health reaches 0, the lair is destroyed and drops loot
5. **Mobs**: Click on mobs to damage them

## Customization

### Adjust Spawn Rates

Edit the `LAIR_TYPES` in `src/objects/MobLair.tsx`:

```typescript
goblin_camp: {
  spawnInterval: 15000, // Change from 30000 to 15000 (spawn twice as fast)
  maxMobs: 16, // Double the mob limit
  // ... other properties
}
```

### Add New Lair Types

```typescript
// In src/objects/MobLair.tsx
vampire_castle: {
  type: 'vampire_castle',
  maxHealth: 2000,
  spawnInterval: 45000,
  mobType: 'vampire',
  maxMobs: 6,
  lootTable: [
    { item: 'blood_ruby', quantity: 3, dropChance: 0.8 },
  ],
}
```

Don't forget to also add it to the server in `server/GameServer.js`!

## Testing

### Test Mob Spawning

1. Create a goblin camp
2. Wait 30 seconds
3. You should see a goblin spawn near the camp
4. After spawning 8 goblins, it will stop until some are killed

### Test Lair Destruction

1. Create any lair
2. Click on it multiple times to damage it
3. When health reaches 0, it should:
   - Stop spawning mobs
   - Display a skull icon
   - Drop loot at its location

### Test Multiplayer Sync

1. Start the server
2. Open the game in two browser tabs
3. Create a lair in one tab
4. You should see it appear in the other tab
5. Damage it in one tab, health should update in both

## Common Issues

### Server won't start
- Make sure port 3001 is not in use
- Check that you ran `npm install` in the server directory

### Mobs not spawning
- Wait for the spawn interval (30-120 seconds depending on lair type)
- Check if mob limit is reached (hover over lair to see count)
- Verify the lair is not destroyed

### Changes not syncing between tabs
- Make sure the server is running
- Check browser console for WebSocket errors
- Verify you're connecting to the correct server URL

## Next Steps

- Read the full documentation: `src/README.md`
- See implementation details: `MOB_LAIR_IMPLEMENTATION.md`
- Check out the examples: `src/examples/`
- Run the tests: `cd src && npm test`

## Architecture Overview

```
┌─────────────────────┐
│   React Client      │
│   - GameWorld       │
│   - MobLair         │
│   - MobStore        │
└──────────┬──────────┘
           │ WebSocket
           │
┌──────────┴──────────┐
│   Game Server       │
│   - Express API     │
│   - Socket.IO       │
│   - Spawn Timers    │
└─────────────────────┘
```

## Key Files

- `src/objects/MobLair.tsx` - Lair component
- `src/store/mobStore.ts` - State management
- `src/components/GameWorld.tsx` - Main game container
- `server/GameServer.js` - Multiplayer server
- `src/styles/game.css` - Styling

## Support

For issues or questions:
1. Check the documentation in `src/README.md`
2. Review `MOB_LAIR_IMPLEMENTATION.md`
3. Open an issue on GitHub

Happy gaming! 🎮
