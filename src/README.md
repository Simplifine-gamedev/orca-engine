# Orca RTS - Scout Unit Implementation

This directory contains the implementation of the scout unit system for Orca RTS, addressing feature request ORC-166.

## Overview

Scout units are specialized reconnaissance units designed for early game map exploration. They feature:

- **Fast Movement Speed** (8.0-9.0) - Faster than combat units
- **Large Vision Range** (10-14) - Extended sight for map exploration  
- **Low Cost** (45-60 resources) - Affordable in early game
- **Low Combat Capability** (3-8 attack) - Not designed for fighting
- **Early Availability** - Can be built from town center or faction-specific buildings

## Project Structure

```
src/
├── config/
│   ├── factions.ts          # Base faction and unit configurations
│   └── factionLoader.ts     # Dynamic loader for faction-specific configs
├── store/
│   └── gameStore.ts         # Game state management and unit creation
├── tests/
│   └── test_scout_units.ts  # Test suite for scout functionality
├── package.json             # NPM package configuration
├── tsconfig.json            # TypeScript compiler configuration
└── README.md                # This file

generated_factions/
├── humans_scout.json        # Human faction scout unit
├── elves_scout.json         # Elf faction scout unit  
├── orcs_scout.json          # Orc faction scout unit
├── undead_scout.json        # Undead faction scout unit
└── README.md                # Faction-specific scout documentation
```

## Scout Units by Faction

### Human Scout
- Balanced, reliable reconnaissance
- Special: Keen Sight (detects hidden units)
- Best for: General purpose exploration

### Elf Ranger  
- Fastest scout with best vision
- Special: Forest Stealth (invisible in forests)
- Best for: Forest maps and stealth missions

### Orc Warg Rider
- Most durable scout with combat capability
- Special: Intimidate and Savage Charge
- Best for: Aggressive scouting and harassment

### Undead Shade
- Cheapest scout with phase walking
- Special: Phase Walk (move through obstacles)
- Best for: Infiltration and difficult terrain

## Usage

### Basic Setup

```typescript
import { scoutUnitConfig, getUnitConfig } from './config/factions';
import gameStore from './store/gameStore';

// Initialize game
gameStore.reset();
gameStore.addPlayer('player1', 'Player Name', 'base');

// Create a scout unit
const scout = gameStore.createUnit('player1', 'scout', { x: 100, y: 100 });
```

### Using Faction Loader

```typescript
import { getFactionLoader } from './config/factionLoader';

const loader = getFactionLoader('./generated_factions');

// Get all scout units
const scouts = loader.getScoutUnits();

// Get faction-specific units
const elfUnits = loader.getUnitsByFaction('elves');

// Get units from building
const townCenterUnits = loader.getUnitsFromBuilding('town_center');
```

### Creating Units in Game

```typescript
// Check if player can afford unit
const canAfford = gameStore.canAffordUnit('player1', scoutUnitConfig);

if (canAfford) {
  // Create the unit
  const unit = gameStore.createUnit('player1', 'scout', { x: 100, y: 100 });
  
  // Select and move the unit
  gameStore.selectUnits([unit.id]);
  gameStore.moveUnit(unit.id, { x: 200, y: 200 });
}
```

## Installation

```bash
cd src
npm install
npm run build
```

## Running Tests

```bash
npm test
```

Expected output:
- Scout configuration validation
- Unit retrieval by ID
- Building availability checks
- Player and unit creation
- Resource management
- Unit selection
- Insufficient resources handling

## Integration with Game Engine

To integrate these scout units with the Godot game engine:

1. **Load configurations**: Import JSON files at game startup
2. **Create unit scenes**: Build Godot scenes for each scout type
3. **Implement movement**: Use scout's `movementSpeed` for character movement
4. **Implement vision**: Use `visionRange` for fog of war revealing
5. **Add special abilities**: Implement faction-specific abilities
6. **UI integration**: Add scout icons to build menus

## Design Philosophy

Scout units follow these design principles:

1. **Early Game Focus**: Available immediately from starting buildings
2. **Exploration Reward**: Large vision range encourages map exploration
3. **Risk vs Reward**: Fragile units that require careful positioning
4. **Faction Identity**: Each faction's scout reflects their playstyle
5. **Resource Efficiency**: Low cost enables multiple scouts

## Balance Considerations

- Scouts should cost roughly 1/3 of a basic combat unit
- Movement speed should be ~1.5-2x faster than infantry
- Vision range should be 2-3x larger than combat units
- Health should be low enough that combat units kill them easily
- Attack power should be minimal (losing to workers in direct combat)

## Future Enhancements

Potential improvements for scout units:

- [ ] Veterancy system (scouts gain XP from exploration)
- [ ] Camouflage mechanics (reduce detection range)
- [ ] Signal flare ability (reveal area for allies)
- [ ] Waypoint patrol system
- [ ] Auto-explore command
- [ ] Scout towers/outposts
- [ ] Mounted vs foot scout variants

## API Reference

### UnitConfig Interface

```typescript
interface UnitConfig {
  id: string;
  name: string;
  type: 'scout' | 'warrior' | 'worker' | 'siege';
  cost: { gold?: number; wood?: number; food?: number; };
  stats: {
    health: number;
    attack: number;
    defense: number;
    movementSpeed: number;
    visionRange: number;
  };
  buildTime: number;
  availableFrom: string[];
  description: string;
}
```

### Game Store Methods

- `createUnit(playerId, unitConfigId, position)` - Create a new unit
- `canAffordUnit(playerId, unitConfig)` - Check resource availability
- `getPlayerUnits(playerId)` - Get all units for a player
- `selectUnits(unitIds)` - Select multiple units
- `moveUnit(unitId, targetPosition)` - Move a unit

## Contributing

When adding new scout units:

1. Create JSON config in `generated_factions/`
2. Follow existing naming convention: `{faction}_scout.json`
3. Ensure stats follow balance guidelines
4. Add special abilities that fit faction theme
5. Update tests to include new scout

## License

MIT License - See main project LICENSE file

## Credits

Implemented in response to user feedback:
- Gaudio: "scout unit"
- Haridzieko: "maybe add a scout unit for this as well"

Issue: ORC-166 - Add scout unit for early game exploration
