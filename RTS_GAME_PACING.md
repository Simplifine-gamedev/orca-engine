# RTS Game - Early Game Pacing Improvements (ORC-168)

## Problem Statement

Early game pacing was too slow with insufficient starting resources and limited gameplay options. Players reported:
- Food/resources came too slowly
- Not enough activities in the early game
- Lack of exploration mechanics

## Implemented Solutions

### 1. Increased Starting Resources
**Location**: `src/store/gameStore.ts` & `server/GameServer.js`

Starting resources have been **doubled** to give players more initial options:

| Resource | Original | New Value | Increase |
|----------|----------|-----------|----------|
| Food     | 200      | **400**   | +100%    |
| Wood     | 150      | **300**   | +100%    |
| Stone    | 100      | **200**   | +100%    |
| Gold     | 50       | **100**   | +100%    |

### 2. Faster Resource Gathering
**Location**: `src/store/gameStore.ts` & `server/GameServer.js`

Resource gathering rates increased by **50%** to speed up early economy:

| Resource | Original Rate | New Rate | Increase |
|----------|--------------|----------|----------|
| Food     | 10/sec       | **15/sec** | +50%   |
| Wood     | 8/sec        | **12/sec** | +50%   |
| Stone    | 6/sec        | **9/sec**  | +50%   |
| Gold     | 4/sec        | **6/sec**  | +50%   |

### 3. New Scout Unit
**Location**: `src/store/gameStore.ts` & `server/GameServer.js`

Added a dedicated **Scout** unit for early exploration:

**Scout Stats:**
- Cost: 50 food (very affordable)
- Build Time: 5 seconds (very fast)
- Speed: 8 (fastest unit)
- Vision Range: 12 (largest in game)
- Health: 75
- Purpose: Early game exploration and map visibility

This gives players an immediate activity while waiting for resources to accumulate.

### 4. Early Game Quest System
**Location**: `src/store/gameStore.ts` & `server/GameServer.js`

Implemented **5 progressive quests** to guide early gameplay and provide goals:

#### Tutorial 1: First Steps
- **Objective**: Train your first scout
- **Rewards**: 100 food, 50 wood

#### Tutorial 2: Gather Resources
- **Objective**: Collect 500 food
- **Rewards**: 100 wood, 50 stone, 25 gold

#### Tutorial 3: Build Your Base
- **Objective**: Construct your first house
- **Rewards**: 150 food, 100 wood, 50 stone

#### Tutorial 4: Expand Economy
- **Objective**: Build a farm
- **Rewards**: 200 food, 150 wood, 100 stone, 50 gold

#### Tutorial 5: Train Your Army
- **Objective**: Build a barracks and train 2 soldiers
- **Rewards**: 300 food, 200 wood, 150 stone, 100 gold

**Total Potential Quest Rewards**: 950 food, 600 wood, 450 stone, 250 gold

### 5. Reduced Build Times
**Location**: `src/store/gameStore.ts` & `server/GameServer.js`

All early buildings and units have **significantly reduced build times**:

#### Units
| Unit    | Original | New Time | Reduction |
|---------|----------|----------|-----------|
| Scout   | N/A      | **5s**   | New unit  |
| Worker  | 15s      | **10s**  | -33%      |
| Soldier | 20s      | **12s**  | -40%      |
| Archer  | 25s      | **15s**  | -40%      |

#### Buildings
| Building    | Original | New Time | Reduction |
|-------------|----------|----------|-----------|
| House       | 30s      | **15s**  | -50%      |
| Barracks    | 45s      | **25s**  | -44%      |
| Farm        | 35s      | **20s**  | -43%      |
| Lumber Mill | 40s      | **25s**  | -38%      |
| Stone Mine  | 50s      | **30s**  | -40%      |

### 6. Additional Starting Units
**Location**: `src/store/gameStore.ts` & `server/GameServer.js`

Players now start with:
- **3 Workers** (increased from 1)
- 1 Town Center

This allows for immediate multi-tasking and faster economic development.

## Impact on Gameplay

### Before
- Start with 1 worker, 200 food, 150 wood
- Slow resource gathering (10 food/sec)
- 30-50 second build times for first buildings
- Limited options in first 2-3 minutes
- No exploration mechanics

### After
- Start with **3 workers**, **400 food**, **300 wood**
- Fast resource gathering (**15 food/sec**, +50%)
- **15-25 second** build times for first buildings
- **Scout unit** available immediately for exploration
- **5 progressive quests** providing goals and bonus resources
- More resources available for experimentation

## Testing the Changes

To test the game server with new pacing:

```bash
cd server
node GameServer.js
```

This will run a demo showing:
1. Improved starting resources
2. Training a scout unit
3. Building construction with reduced times
4. Quest completion and rewards

## Files Modified

1. `src/store/gameStore.ts` - Game state management and configuration
2. `server/GameServer.js` - Server-side game logic and coordination

## Next Steps (Future Improvements)

1. Balance testing with real players
2. Add more varied early quests
3. Implement scout special abilities (e.g., reveal map fog)
4. Add tutorial tooltips in UI
5. Consider adding an "aggressive start" game mode option
6. Telemetry to track early game engagement metrics

## References

- Linear Issue: **ORC-168**
- User Feedback: Haridzieko - "pacing was slow in the beginning again, food comes slow, not much to do"
