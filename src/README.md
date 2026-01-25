# Orca RTS Game

This directory contains the TypeScript/React implementation for the Orca RTS game.

## Project Structure

```
src/
├── buildings/          # Building components and models
│   ├── Building.tsx    # Main building component with construction & training logic
│   └── buildingModels.ts  # Building configurations and models
├── config/             # Game configuration files
│   └── factions.ts     # Faction and unit configurations
└── units/              # Unit components (to be implemented)
```

## Buildings

### Archery Range

The **Archery Range** is a military building that trains ranged units.

#### Configuration

- **Cost**: 175 wood, 25 stone
- **Build Time**: 70 seconds
- **Health**: 1200 HP
- **Armor**: 2
- **Size**: 80x80
- **Color**: #6b8e23 (olive green)

#### Trainable Units

1. **Archer**
   - Basic ranged unit
   - Cost: 50 food, 40 wood
   - Train time: 22 seconds
   - Health: 60 HP
   - Attack: 10 (12 for Elves)
   - Range: 6 (7 for Elves)
   - Speed: 1.1

2. **Crossbowman**
   - Advanced ranged unit
   - Cost: 70 food, 50 wood, 20 gold
   - Train time: 30 seconds
   - Health: 70 HP
   - Attack: 16 (18 for Elves)
   - Range: 7 (8 for Elves)
   - Speed: 1.0

#### Requirements

- Town Hall must be constructed before building an Archery Range

## Features Implemented

### Building Component (`Building.tsx`)

The Building component handles:

- ✅ Construction progress tracking
- ✅ Unit training queue management
- ✅ Visual feedback for construction and training
- ✅ Preview mode for building placement
- ✅ Click handling and selection
- ✅ Health and state management
- ✅ Proper model/visual rendering

### Building Models (`buildingModels.ts`)

Includes configurations for:

- ✅ Town Hall
- ✅ Barracks
- ✅ **Archery Range** (fixed)
- ✅ Stable
- ✅ Blacksmith
- ✅ Guard Tower
- ✅ Farm
- ✅ Lumber Mill
- ✅ Stone Mine
- ✅ Market

Utility functions:
- `getBuildingModel(type)` - Get building configuration
- `getBuildingsByType(category)` - Filter buildings by category
- `canBuild(type, buildings, resources)` - Check if building can be constructed

### Factions (`factions.ts`)

Four playable factions with unique bonuses:

1. **Human Alliance** - Balanced, versatile units
2. **Elven Kingdom** - Superior archers and ranged combat
3. **Orcish Horde** - Powerful melee units
4. **Undead Legion** - Cheap, fast-training units

Each faction has access to the Archery Range and can train archers/crossbowmen with faction-specific bonuses.

#### Elven Kingdom - Archery Specialists

The Elves have special bonuses for archery:
- Archers and crossbowmen have +2 range and +2 attack
- Archery range trains units 20% faster
- Enhanced archer unit stats by default

## Fixed Issues

### Archery Range (Issue ORC-150)

**Status**: ✅ FIXED

**Changes Made**:
1. Created proper building model configuration in `buildingModels.ts`
2. Set appropriate costs, stats, and trainable units
3. Configured archer and crossbowman unit definitions
4. Integrated with all four factions
5. Added Elven faction bonuses for archery units
6. Implemented training queue and progress tracking in Building component

**Unit Training**: Fixed in this implementation with proper queue management and progress tracking.

**Model/Preview**: Properly configured with appropriate size, color, and visual feedback for construction/training states.

## Usage Example

```typescript
import { Building } from './buildings/Building';
import { buildingModels } from './buildings/buildingModels';
import { factions } from './config/factions';

// Create an archery range
<Building
  id="archery_1"
  type="archery_range"
  position={{ x: 100, y: 100 }}
  faction="elves"
  onSelect={(id) => console.log('Selected:', id)}
  onComplete={(id) => console.log('Construction complete:', id)}
/>

// Check if we can build
const { canBuild, reason } = canBuild(
  'archery_range',
  ['town_hall'],
  { wood: 200, stone: 30, gold: 0 }
);

// Get archery range config
const archeryConfig = buildingModels.archery_range;
console.log(archeryConfig.trainableUnits); // ['archer', 'crossbowman']
```

## Next Steps

- [ ] Implement 3D models for archery range
- [ ] Add proper sprites/textures
- [ ] Implement archer and crossbowman unit components
- [ ] Add sound effects for training
- [ ] Implement unit upgrades at blacksmith
- [ ] Add range indicators for ranged units
