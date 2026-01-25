# Orca RTS Game

A real-time strategy game built on the Orca engine with TypeScript configuration.

## Features

### Units

#### Scout Unit (NEW - ORC-147)
The scout unit is designed for early game exploration with the following characteristics:

- **Cost**: 50 gold, 25 food (very affordable for early game)
- **Stats**:
  - Health: 60 HP (low survivability)
  - Attack: 5 (minimal combat ability)
  - Defense: 2 (light armor)
  - Movement Speed: 8.0 (fastest early game unit)
  - Vision Range: 15 (excellent for exploration)
- **Build Time**: 15 seconds (quick production)
- **Available From**: Town Center, Stable

**Gameplay Role**: 
- Primary: Map exploration and reconnaissance
- Secondary: Early game harassment and hit-and-run tactics
- Excellent for discovering resources, enemy positions, and map layout
- Not meant for direct combat - use speed to escape threats

#### Other Units
- **Warrior**: Balanced melee unit from Barracks
- **Archer**: Ranged unit from Archery Range

## File Structure

```
src/
  config/
    factions.ts         - Faction and unit configurations
  store/
    gameStore.ts        - Game state management
generated_factions/
  scout.json            - Scout unit detailed config
  default_faction.json  - Default faction setup
```

## Usage

### Creating a Scout Unit

```typescript
import { gameStore } from './store/gameStore';
import { scoutUnit } from './config/factions';

// Initialize game
gameStore.initializeGame(1);

// Create a scout at position (100, 100)
const scoutId = gameStore.createScout('player_0', { x: 100, y: 100 });

// Or use the generic createUnit method
const scoutId2 = gameStore.createUnit('player_0', scoutUnit, { x: 150, y: 150 });

// Get visible area for the scout
const visibleArea = gameStore.getVisibleArea(scoutId);
console.log(`Scout can see ${visibleArea.radius} units in all directions`);
```

### Checking Unit Availability

```typescript
import { getUnitsFromBuilding } from './config/factions';

// Get all units that can be built from town center
const townCenterUnits = getUnitsFromBuilding('town_center');
console.log('Town Center can produce:', townCenterUnits.map(u => u.name));
// Output: Town Center can produce: ["Scout"]

// Get all units from stable
const stableUnits = getUnitsFromBuilding('stable');
console.log('Stable can produce:', stableUnits.map(u => u.name));
// Output: Stable can produce: ["Scout"]
```

### Resource Management

```typescript
// Check if player can afford a scout
const player = gameStore.getState().players.get('player_0');
const resources = gameStore.getPlayerResources('player_0');
console.log('Current resources:', resources);
// { gold: 500, wood: 300, food: 200 }

// Create scout (costs 50 gold, 25 food)
gameStore.createScout('player_0', { x: 200, y: 200 });

// Check resources after purchase
console.log('After scout:', gameStore.getPlayerResources('player_0'));
// { gold: 450, wood: 300, food: 175 }
```

## Configuration

### Unit Stats Reference

| Stat | Scout | Warrior | Archer |
|------|-------|---------|--------|
| Health | 60 | 150 | 80 |
| Attack | 5 | 15 | 12 |
| Defense | 2 | 10 | 5 |
| Speed | 8.0 | 3.5 | 4.0 |
| Vision | 15 | 8 | 10 |
| Cost (Gold) | 50 | 100 | 80 |
| Cost (Food) | 25 | 50 | - |
| Cost (Wood) | - | - | 40 |

### Buildings

- **Town Center**: Produces scouts, serves as main base
- **Barracks**: Produces warriors
- **Archery Range**: Produces archers
- **Stable**: Produces scouts (alternative to town center)

## Development

### Adding New Units

1. Define unit config in `src/config/factions.ts`:

```typescript
export const newUnit: UnitConfig = {
  id: "cavalry",
  name: "Cavalry",
  cost: { gold: 120, food: 60 },
  stats: {
    health: 200,
    attack: 20,
    defense: 8,
    movementSpeed: 6.5,
    visionRange: 10,
  },
  buildTime: 40,
  availableFrom: ["stable"],
  description: "Heavy mounted unit.",
};
```

2. Add to faction's units array
3. Create JSON config in `generated_factions/`
4. Update game store with any special unit creation methods if needed

### Testing

```typescript
// Initialize test game
gameStore.initializeGame(2);

// Test scout creation
const scout = gameStore.createScout('player_0', { x: 0, y: 0 });
console.assert(scout !== null, 'Scout should be created');

// Test movement
const moved = gameStore.moveUnit(scout, { x: 50, y: 50 });
console.assert(moved, 'Scout should move successfully');

// Test vision
const vision = gameStore.getVisibleArea(scout);
console.assert(vision.radius === 15, 'Scout vision range should be 15');
```

## User Feedback Implementation

This feature implements user feedback from:
- **Gaudio**: "scout unit"
- **Haridzieko**: "maybe add a scout unit for this as well" (for early game pacing)

The scout unit addresses early game pacing by:
1. Being affordable (50 gold, 25 food)
2. Building quickly (15 seconds)
3. Moving fast (8.0 speed)
4. Providing excellent vision (15 range)
5. Being available from town center immediately

## Future Enhancements

- Add scout upgrades (improved vision, speed)
- Add stealth/camouflage abilities
- Add scout special abilities (flares, tracking)
- Add mounted vs. foot scout variants
- Add faction-specific scout variations
