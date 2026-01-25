# Orca RTS Game

A real-time strategy game built on the Orca Engine.

## Overview

Orca RTS is a web-based real-time strategy game featuring resource gathering, base building, and unit management. The game emphasizes strategic decision-making and efficient resource management.

## Recent Changes

### ORC-145: Early Game Pacing Improvements

Based on user feedback (Haridzieko: "pacing was slow in the beginning again, food comes slow, not much to do"), we've implemented significant improvements to early game pacing:

#### 1. **Increased Starting Resources** ✅
- **Gold**: 100 → 200 (+100%)
- **Wood**: 100 → 200 (+100%)
- **Food**: 100 → 150 (+50%)
- **Stone**: 50 → 100 (+100%)

*Rationale: Players now have more resources to work with from the start, allowing for faster initial building and unit production.*

#### 2. **Faster Resource Gathering** ✅
- **Villager gathering rates doubled** from 0.5 to 1.0 resources/second
- **Food gathering increased by 140%** (0.5 → 1.2 per second)
- **Gold & Wood gathering doubled** (0.5 → 1.0 per second)
- **Stone gathering increased by 60%** (0.5 → 0.8 per second)

*Rationale: Resources accumulate faster, reducing wait times and increasing early game activity.*

#### 3. **Scout Unit Added** ✅
- **New unit type**: Scout
- **Starting units**: 1 scout included at game start
- **Stats**:
  - Health: 60
  - Speed: 2.5 (fastest unit)
  - Vision radius: 150 (large)
  - Build time: 15 seconds
  - Cost: 40 gold, 15 food

*Rationale: Provides early game exploration and scouting opportunities, giving players more to do while waiting for resources.*

#### 4. **Reduced Build Times** ✅
Early game buildings build 40-50% faster:
- **Barracks**: 30s → 15s (-50%)
- **Farm**: 20s → 10s (-50%)
- **Lumber Mill**: 25s → 12s (-52%)
- **Mining Camp**: 25s → 12s (-52%)
- **House**: 15s → 8s (-47%)

*Rationale: Faster building allows players to expand their economy quicker and reach mid-game faster.*

#### 5. **Increased Starting Units** ✅
- **Villagers**: 3 → 4 (+1)
- **Scouts**: 0 → 1 (+1)
- **Total starting population**: 3 → 5

*Rationale: More units at the start means more simultaneous actions, reducing early game downtime.*

## Game Structure

```
rts-game/
├── src/
│   ├── store/
│   │   └── gameStore.ts        # Client-side game state management
│   ├── units/                  # Unit definitions and logic
│   └── buildings/              # Building definitions and logic
├── server/
│   └── GameServer.js           # Server-side game logic
├── gameConfig.json             # Central configuration file
└── README.md                   # This file
```

## Configuration

All game parameters are centralized in `gameConfig.json` for easy balancing and adjustments. Key configuration sections:

- **Starting Resources**: Initial player resources
- **Starting Units**: Units provided at game start
- **Gathering Rates**: Resource collection speeds per unit type
- **Build Times**: Construction durations for buildings
- **Unit Costs & Stats**: Unit properties and costs
- **Building Costs & Stats**: Building properties and costs

## Files Modified for ORC-145

### 1. `src/store/gameStore.ts`
- Updated `STARTING_RESOURCES` constants
- Updated `GATHERING_RATES` for all unit types
- Updated `BUILD_TIMES` for all buildings
- Added scout unit to `UNIT_COSTS` and `UNIT_STATS`
- Updated `STARTING_UNITS` configuration
- Added scout creation logic in `createStartingUnits()`

### 2. `server/GameServer.js`
- Updated `getInitialGameState()` with new starting resources
- Updated `createStartingUnits()` to include scout unit
- Added scout unit to `getUnitCost()` method
- Updated `getBuildTime()` with reduced build times
- Updated `getGatheringRate()` with increased rates
- Added scout stats to helper methods

### 3. `gameConfig.json` (New)
- Created centralized configuration file
- Documented all pacing improvements
- Listed all game parameters in one place

## Unit Types

### Villager
- **Role**: Primary economic unit
- **Health**: 50
- **Speed**: 1.2
- **Gathering Rate**: 1.0-1.2 (varies by resource)
- **Cost**: 50 gold, 25 food
- **Build Time**: 20 seconds

### Scout (NEW - ORC-145)
- **Role**: Fast exploration and map vision
- **Health**: 60
- **Speed**: 2.5 (fastest)
- **Vision Radius**: 150 (largest)
- **Gathering Rate**: 0.5-0.8 (can gather but slower)
- **Cost**: 40 gold, 15 food
- **Build Time**: 15 seconds

### Warrior
- **Role**: Melee combat unit
- **Health**: 100
- **Speed**: 1.5
- **Cost**: 60 gold, 40 food
- **Build Time**: 30 seconds

### Archer
- **Role**: Ranged combat unit
- **Health**: 70
- **Speed**: 1.3
- **Cost**: 45 gold, 35 food, 20 wood
- **Build Time**: 25 seconds

## Building Types

### Town Center
- **Role**: Main building, trains villagers
- **Health**: 1000
- **Cost**: 500 wood, 300 stone
- **Build Time**: 60 seconds

### Barracks
- **Role**: Trains military units
- **Health**: 500
- **Cost**: 150 wood, 50 gold
- **Build Time**: 15 seconds (reduced from 30s)

### Farm
- **Role**: Generates food over time
- **Health**: 200
- **Cost**: 60 wood
- **Build Time**: 10 seconds (reduced from 20s)

### Lumber Mill
- **Role**: Improves wood gathering
- **Health**: 300
- **Cost**: 100 wood, 50 gold
- **Build Time**: 12 seconds (reduced from 25s)

### Mining Camp
- **Role**: Improves gold and stone gathering
- **Health**: 300
- **Cost**: 100 wood, 50 gold
- **Build Time**: 12 seconds (reduced from 25s)

### House
- **Role**: Increases population cap
- **Health**: 250
- **Cost**: 50 wood
- **Build Time**: 8 seconds (reduced from 15s)

## Usage

### Client-Side (TypeScript)

```typescript
import { gameStore } from './src/store/gameStore';

// Get current game state
const state = gameStore.getState();

// Subscribe to state changes
const unsubscribe = gameStore.subscribe((newState) => {
  console.log('Game state updated:', newState);
});

// Gather resources
gameStore.gatherResource('food', 1.2); // Food gathering rate

// Train a scout unit (NEW!)
const scoutCost = UNIT_COSTS.scout; // { gold: 40, food: 15 }
if (gameStore.spendResources(scoutCost)) {
  const scout = {
    id: 'scout-1',
    type: 'scout',
    position: { x: 150, y: 150 },
    health: 60,
    maxHealth: 60,
    speed: 2.5,
  };
  gameStore.addUnit(scout);
}
```

### Server-Side (JavaScript)

```javascript
const GameServer = require('./server/GameServer');

const gameServer = new GameServer();

// Create a new game
const game = gameServer.createGame('game-123', 'player-1');

// Start the game
gameServer.startGame('game-123');

// Train a scout unit
gameServer.trainUnit('game-123', 'player-1', 'scout', 'town-center-0');

// Listen to game events
gameServer.on('game:update', ({ gameId, state }) => {
  console.log(`Game ${gameId} updated:`, state);
});

// Construct a building with reduced build time
gameServer.constructBuilding('game-123', 'player-1', 'barracks', { x: 300, y: 200 });
// Barracks will complete in 15 seconds instead of 30 seconds!
```

## Testing the Pacing Improvements

To experience the pacing improvements:

1. **Start a new game** and observe the increased starting resources
2. **Send villagers to gather** and notice the faster accumulation
3. **Train the starting scout** and use it to explore the map
4. **Build a barracks** and notice it completes in 15 seconds
5. **Build farms** to boost food production (only 10 seconds each)

### Before vs After Comparison

| Metric | Before (Old) | After (ORC-145) | Change |
|--------|--------------|-----------------|--------|
| Starting Gold | 100 | 200 | +100% |
| Starting Wood | 100 | 200 | +100% |
| Starting Food | 100 | 150 | +50% |
| Starting Stone | 50 | 100 | +100% |
| Villager Gather Rate | 0.5/s | 1.0-1.2/s | +100-140% |
| Barracks Build Time | 30s | 15s | -50% |
| Farm Build Time | 20s | 10s | -50% |
| Starting Units | 3 villagers | 4 villagers + 1 scout | +67% |
| Scout Unit | N/A | Available | NEW |

## User Feedback

> "pacing was slow in the beginning again, food comes slow, not much to do - maybe add a scout unit for this as well"
> — Haridzieko

**Our Response:**
- ✅ Increased food gathering rate by 140%
- ✅ Added scout unit for early exploration
- ✅ Increased overall starting resources and gathering rates
- ✅ Reduced build times to provide faster progression
- ✅ Added extra starting villager for more simultaneous actions

## Future Improvements

Potential additional enhancements for game pacing:

1. **Early Game Quests/Objectives**
   - Tutorial missions to guide new players
   - Optional objectives for bonus resources

2. **Resource Deposit Proximity**
   - Spawn initial resources closer to starting position
   - Reduce travel time for early gathering

3. **Population Cap Adjustments**
   - Increase starting population cap to 15
   - Allow earlier expansion

4. **Fast Start Game Mode**
   - Optional game mode with even more starting resources
   - For players who prefer mid-game action

## License

MIT License - Part of the Orca Engine project

## Contributing

When adjusting game balance, please update:
1. `gameConfig.json` - Central configuration
2. `src/store/gameStore.ts` - Client-side constants
3. `server/GameServer.js` - Server-side logic
4. This README - Documentation

---

**Issue Resolved**: ORC-145 - Early game pacing improvements  
**Date**: 2026-01-25  
**Status**: ✅ Complete
