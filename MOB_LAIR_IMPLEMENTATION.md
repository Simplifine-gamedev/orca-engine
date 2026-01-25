# Mob Lair Implementation - ORC-185

This document describes the implementation of mob lair spawning mechanics for Orca RTS.

## Overview

The mob lair system allows for continuous mob spawning from designated lair structures. Players can destroy lairs to stop mob spawning and receive loot rewards.

## Implementation Summary

### Files Created

1. **`src/objects/MobLair.tsx`** - Main lair component
   - Handles mob spawning logic
   - Manages lair health and destruction
   - Implements loot generation
   - Supports 4 lair types: goblin camp, ogre cave, undead crypt, dragon nest

2. **`src/store/mobStore.ts`** - State management
   - Zustand store for lairs and mobs
   - CRUD operations for lairs and mobs
   - Efficient state updates and queries
   - Selectors for performance optimization

3. **`server/GameServer.js`** - Multiplayer server
   - Express + Socket.IO implementation
   - Real-time mob spawning across clients
   - REST API for lair management
   - Game loop for mob AI
   - Automatic cleanup of destroyed entities

4. **`src/components/GameWorld.tsx`** - Game container
   - Integrates lairs and mobs
   - Handles client-server communication
   - Manages game state synchronization
   - Provides UI for lair creation

5. **Supporting Files**
   - `src/styles/game.css` - Styling for game elements
   - `src/examples/BasicExample.tsx` - Simple usage example
   - `src/examples/CustomLairExample.tsx` - Advanced usage example
   - `src/README.md` - Comprehensive documentation
   - `server/package.json` - Server dependencies
   - `src/package.json` - Client dependencies

## Features Implemented

### ✅ Core Features

- [x] Multiple lair types (4 types implemented)
- [x] Periodic mob spawning with configurable intervals
- [x] Lair health system
- [x] Lair destruction stops spawning
- [x] Loot drops with randomized quantities
- [x] Different loot tables per lair type

### ✅ Technical Features

- [x] Real-time multiplayer synchronization
- [x] State management with Zustand
- [x] WebSocket communication (Socket.IO)
- [x] REST API for lair management
- [x] Automatic mob cleanup
- [x] Efficient rendering with React
- [x] Type-safe TypeScript implementation

### ✅ Game Mechanics

- [x] Max mob limit per lair
- [x] Spawn radius around lair
- [x] Health bars for lairs and mobs
- [x] Click-to-damage interaction
- [x] Visual feedback for destruction
- [x] Mob count display on lairs

## Lair Types Specification

### 1. Goblin Camp
- **Health**: 500
- **Spawn Interval**: 30 seconds
- **Max Mobs**: 8 goblins
- **Loot**: Gold (50), Goblin Dagger (30% chance), Leather Scraps (70% chance)

### 2. Ogre Cave
- **Health**: 1200
- **Spawn Interval**: 60 seconds
- **Max Mobs**: 4 ogres
- **Loot**: Gold (150), Ogre Club (40% chance), Thick Hide (80% chance)

### 3. Undead Crypt
- **Health**: 800
- **Spawn Interval**: 25 seconds
- **Max Mobs**: 12 skeletons
- **Loot**: Gold (80), Bone Sword (25% chance), Soul Essence (60% chance)

### 4. Dragon Nest
- **Health**: 3000
- **Spawn Interval**: 120 seconds
- **Max Mobs**: 2 dragons
- **Loot**: Gold (500), Dragon Scale (90% chance), Legendary Weapon (10% chance)

## Architecture

### Client-Server Communication

```
Client (React)                Server (Node.js)
     │                              │
     ├──[WebSocket Connect]─────────>
     │                              │
     <─────[game:state]─────────────┤ (Initial sync)
     │                              │
     ├────[lair:damage]────────────>
     │                              │
     <─────[lair:damaged]───────────┤ (Broadcast)
     │                              │
     <─────[mob:spawned]────────────┤ (Auto spawn)
     │                              │
     ├────[mob:damage]─────────────>
     │                              │
     <─────[mob:killed]─────────────┤ (Broadcast)
```

### State Management Flow

```
Action → Store Update → React Re-render
   ↓
Socket Emit → Server → Broadcast → Other Clients
```

## Usage

### Quick Start

1. **Install dependencies**:
   ```bash
   # Server
   cd server
   npm install
   
   # Client
   cd ../src
   npm install
   ```

2. **Start the server**:
   ```bash
   cd server
   npm start
   ```

3. **Use in your app**:
   ```tsx
   import { GameWorld } from './src/components/GameWorld';
   
   function App() {
     return <GameWorld serverUrl="http://localhost:3001" />;
   }
   ```

### Creating Lairs Programmatically

```typescript
import { useMobStore } from './store/mobStore';
import { LAIR_TYPES } from './objects/MobLair';

const { addLair } = useMobStore();

addLair({
  id: 'my_lair_1',
  ...LAIR_TYPES.goblin_camp,
  position: { x: 400, y: 300 },
  health: 500,
  isDestroyed: false,
});
```

### Server API

**Create Lair**:
```bash
curl -X POST http://localhost:3001/api/lairs \
  -H "Content-Type: application/json" \
  -d '{"type": "goblin_camp", "position": {"x": 400, "y": 300}}'
```

**Get Game State**:
```bash
curl http://localhost:3001/api/game-state
```

**Delete Lair**:
```bash
curl -X DELETE http://localhost:3001/api/lairs/LAIR_ID
```

## Testing

### Manual Testing Steps

1. Start the server
2. Open the game in browser
3. Create multiple lairs using UI buttons
4. Verify mobs spawn at correct intervals
5. Click lairs to damage them
6. Verify lairs stop spawning when destroyed
7. Verify loot drops appear
8. Open multiple tabs to test multiplayer sync

### Expected Behavior

- Mobs should spawn within the configured interval
- Spawning should stop when mob limit is reached
- Lairs should be destroyed when health reaches 0
- Loot should drop with correct probabilities
- All changes should sync across connected clients

## Performance Considerations

- **Client-side**: React components re-render only when necessary
- **Server-side**: Game loop runs at 5-second intervals
- **Cleanup**: Dead mobs removed after 2 seconds
- **Lair cleanup**: Destroyed lairs cleaned after 60 seconds
- **Network**: Events broadcast only to relevant clients

## Future Enhancements

Potential improvements for the system:

1. **Advanced AI**
   - Pathfinding for mobs
   - Aggressive/defensive behavior
   - Territory patrol patterns

2. **Lair Mechanics**
   - Lair repair/reconstruction
   - Lair upgrades (health, spawn rate)
   - Boss lair variants
   - Linked lair networks

3. **Combat System**
   - Player attack system
   - Mob attack players
   - Skill-based damage
   - Critical hits

4. **Resource Management**
   - Resource gathering from mobs
   - Lair construction costs
   - Economy system

5. **Visual Polish**
   - Sprite animations
   - Particle effects
   - Sound effects
   - Minimap

## Technical Decisions

### Why Zustand?
- Lightweight and performant
- Simple API without boilerplate
- Great TypeScript support
- Easy to test and debug

### Why Socket.IO?
- Reliable WebSocket with fallbacks
- Built-in room/namespace support
- Automatic reconnection
- Easy event-based communication

### Why Express?
- Minimal and flexible
- Large ecosystem
- Easy to integrate with Socket.IO
- Good for REST APIs

## Troubleshooting

### Mobs not spawning
- Check if lair is destroyed
- Verify mob limit not reached
- Check server console for errors
- Ensure spawn timer is running

### Lairs not syncing
- Verify server is running
- Check WebSocket connection
- Look for CORS issues
- Check browser console

### Performance issues
- Reduce number of active lairs
- Decrease spawn frequency
- Optimize render cycles
- Use React.memo for components

## Contributing

To extend the lair system:

1. Add new lair types to `LAIR_TYPES`
2. Update server-side `LAIR_CONFIGS`
3. Add sprites/icons in helper functions
4. Update documentation
5. Add tests

## License

Part of Orca RTS project - see main project license.

## Contact

For questions or issues related to this implementation, please open an issue on the project repository.

---

**Implementation Date**: January 25, 2026  
**Issue**: ORC-185  
**Status**: ✅ Complete
