# Orca RTS - Economy & Research System

This directory contains the economy, research, and upgrade systems for the Orca RTS game demo.

## Overview

The economy system provides meaningful late-game resource spending options through:

1. **Research Tree** - Technology upgrades across 5 categories
2. **Building Upgrades** - Town center and structure improvements
3. **Unit Upgrades** - Armor, weapons, and stat improvements
4. **Hero Units** - Powerful unique commanders
5. **Special Abilities** - Advanced game mechanics

## Structure

```
src/
├── store/
│   ├── gameStore.ts        # Core game state and resource management
│   └── researchStore.ts    # Research tree and technology system
├── ui/
│   ├── ResearchPanel.tsx          # Research UI component
│   ├── BuildingUpgradePanel.tsx   # Building upgrade UI
│   └── UnitUpgradePanel.tsx       # Unit upgrade and hero UI
└── README.md               # This file
```

## Features

### Game Store (`gameStore.ts`)

- Resource management (gold, wood, stone, food)
- Building management with upgrade system
- Unit management with stat tracking
- Population cap system
- Building unlock progression
- Hero unit creation

### Research Store (`researchStore.ts`)

- 40+ research technologies across 5 categories:
  - **Economy**: Resource gathering improvements
  - **Military**: Combat unit enhancements
  - **Technology**: General improvements
  - **Magic**: Magical abilities and mage units
  - **Special**: Unique game features
- Research prerequisites and tech tree
- Time-based research completion
- Real-time progress tracking

### UI Components

#### Research Panel
- Browse all available research
- Filter by category
- View prerequisites and effects
- Start/cancel research
- Track research progress

#### Building Upgrade Panel
- Upgrade existing buildings
- View upgrade benefits
- Track building levels
- Unlock advanced structures

#### Unit Upgrade Panel
- Upgrade unit stats (armor, weapons, health, speed)
- Create hero units
- View upgrade effects
- Category-based organization

## Usage

### Basic Integration

```typescript
import { gameStore } from './store/gameStore';
import { researchStore } from './store/researchStore';
import ResearchPanel from './ui/ResearchPanel';
import BuildingUpgradePanel from './ui/BuildingUpgradePanel';
import UnitUpgradePanel from './ui/UnitUpgradePanel';

// Subscribe to game state changes
gameStore.subscribe((state) => {
  console.log('Resources:', state.resources);
});

// Subscribe to research updates
researchStore.subscribe((state) => {
  console.log('Completed research:', state.completedResearch.size);
});

// Use UI components
<ResearchPanel isOpen={showResearch} onClose={() => setShowResearch(false)} />
<BuildingUpgradePanel isOpen={showBuildings} onClose={() => setShowBuildings(false)} />
<UnitUpgradePanel isOpen={showUnits} onClose={() => setShowUnits(false)} />
```

### Adding Resources

```typescript
// Add resources from gathering, trading, etc.
gameStore.addResources({ gold: 100, wood: 50 });
```

### Starting Research

```typescript
// Start a research
if (researchStore.canResearch('mining_1')) {
  researchStore.startResearch('mining_1');
}
```

### Upgrading Buildings

```typescript
// Upgrade a building
const building = gameStore.getState().buildings[0];
gameStore.upgradeBuilding(building.id);
```

### Creating Heroes

```typescript
// Create a hero unit (requires Town Center level 5)
gameStore.createHeroUnit(UnitType.HERO_WARRIOR);
```

## Research Tree

### Economy Research
- Mining improvements (3 levels)
- Forestry improvements (2 levels)
- Agriculture improvements (2 levels)
- Trade routes
- Resource conversion

### Military Research
- Infantry combat training (2 levels)
- Archery improvements (2 levels)
- Cavalry training
- Siege engineering
- Fortifications

### Technology Research
- Metallurgy
- Plate armor
- Ballistics
- Engineering
- Architecture

### Magic Research
- Basic magic (unlocks mages)
- Advanced magic
- Arcane mastery
- Divine blessing

### Special Research
- Hero training
- Legendary heroes
- Spy network
- Ancient wonder

## Building Upgrades

### Town Center (5 levels)
- Increases population cap
- Unlocks advanced buildings
- Boosts resource generation
- Level 5 unlocks hero units

### Military Buildings
- Barracks upgrades
- Archery range upgrades
- Stable upgrades

### Support Buildings
- Blacksmith upgrades (better unit upgrades)
- Academy upgrades (magic research)
- Workshop upgrades

## Unit Upgrades

### Infantry
- Armor upgrades (2 levels)
- Weapon upgrades (2 levels)

### Ranged
- Armor upgrades
- Weapon upgrades (2 levels)

### Cavalry
- Armor upgrades
- Weapon upgrades
- Speed upgrades

### Special
- Mage spell power
- Siege weapon damage

## Hero Units

Heroes are powerful unique units that require:
- Town Center level 5
- 2000 gold + 500 food per hero

Available heroes:
- **Hero Warrior**: High HP and melee damage
- **Hero Archer**: High ranged damage
- **Hero Mage**: Powerful magical attacks

## Resource Costs Balance

The system is designed to provide meaningful resource sinks at all game stages:

- **Early game** (0-5 minutes): Basic research and building upgrades
- **Mid game** (5-15 minutes): Advanced research and unit upgrades
- **Late game** (15+ minutes): Hero units, legendary research, and wonder construction

Total possible spending exceeds 100,000 gold equivalent across all systems.

## Future Enhancements

Potential additions:
- Special abilities system
- Wonder construction
- Research queue system
- Building construction
- Advanced hero abilities
- More unit types

## Notes

- All UI components use Tailwind CSS for styling
- State management uses simple pub/sub pattern
- Research timers run in real-time
- Resource checks prevent overspending
- Prerequisites ensure logical progression

## License

Part of the Orca Engine project.
