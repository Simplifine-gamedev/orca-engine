# Orca RTS - Late Game Resource Sinks

This implementation addresses the issue of not having enough things to spend resources on in the late game. It includes a comprehensive research tree, building upgrades, unit upgrades, and various other resource sinks.

## Features Implemented

### 1. Research Tree System (`src/store/researchStore.ts`)
A comprehensive research system with 40+ technologies across 5 categories:

#### Categories:
- **Military**: Unit combat upgrades (melee, ranged, cavalry)
- **Economy**: Resource gathering and capacity improvements
- **Technology**: Building and unit training efficiency
- **Magic**: Spell casters and magical abilities
- **Defense**: Fortifications and defensive structures

#### Key Features:
- Prerequisite system (research must be completed before advancing)
- Real-time progress tracking
- Research queue system
- Category-based organization
- Multi-tier upgrades (Level 1, 2, 3)

### 2. Building System (`src/types/buildings.ts`, `src/store/gameStore.ts`)
10+ building types with upgrade capabilities:

#### Building Types:
- **Town Center**: Population capacity, resource generation (up to Level 5)
- **Barracks**: Train melee units (up to Level 3)
- **Archery Range**: Train ranged units (up to Level 3)
- **Stable**: Train cavalry units (up to Level 3)
- **Workshop**: Train siege units (requires research)
- **Blacksmith**: Unit upgrade discounts (up to Level 3)
- **Academy**: Research speed, mana generation (requires research)
- **Temple**: Mana generation (requires magic research)
- **Market**: Trade efficiency (up to Level 3)
- **Tower**: Defensive structure (requires research)

#### Features:
- Building level progression (most buildings 1-3 levels, Town Center up to 5)
- Prerequisite system (building level or research requirements)
- Unlock system (advanced buildings require research)
- Escalating costs per level (50% increase per level)
- Passive resource generation from buildings

### 3. Unit System (`src/types/units.ts`, `src/store/gameStore.ts`)
8 unit types with extensive upgrade paths:

#### Unit Types:
- **Worker**: Resource gathering
- **Warrior**: Basic melee unit (3 upgrade levels)
- **Archer**: Ranged unit (2 upgrade levels)
- **Cavalry**: Fast melee unit (1 upgrade level)
- **Siege**: Anti-building unit (requires research)
- **Mage**: Magical damage dealer (2 upgrade levels, requires research)
- **Hero**: Powerful unique unit (2 upgrade levels, requires research)
- **Priest**: Support/healing unit (requires research)

#### Features:
- Base stats: HP, attack damage, armor, movement speed, attack speed, range
- Stat upgrade system (increases HP, damage, armor, etc.)
- Special abilities that unlock through research
- Training cost and time
- Prerequisite system (building requirements, research requirements)

### 4. Unit Upgrades
Comprehensive upgrade system for all military units:

#### Upgrade Tiers:
Each military unit has 1-3 upgrade levels with increasing costs:
- **Level 1**: Basic improvements (300-450 gold)
- **Level 2**: Advanced improvements (500-900 gold)
- **Level 3**: Elite improvements (1200+ gold)

#### Stat Bonuses:
- +HP (15-150 depending on unit and level)
- +Attack Damage (3-15)
- +Armor (1-3)
- +Attack Speed (0.1-0.3)
- +Movement Speed (0.2)
- +Attack Range (1-2)

### 5. Special Abilities
Units unlock special abilities through research:

#### Warrior Abilities:
- **Charge**: Rush forward dealing damage (unlocked with melee attack 2)

#### Archer Abilities:
- **Arrow Volley**: Fire multiple arrows in an area (unlocked with ranged attack 2)

#### Cavalry Abilities:
- **Trample**: Charge through enemies (unlocked with cavalry speed)

#### Mage Abilities:
- **Fireball**: Launch a fireball (base ability)
- **Meteor**: Devastating area damage (unlocked with arcane mastery)

#### Hero Abilities:
- **War Cry**: Buff nearby allies
- **Heroic Strike**: Deal massive damage

#### Priest Abilities:
- **Heal**: Restore health to an ally
- **Group Heal**: Heal multiple allies

### 6. Resource Types
5 resource types for diverse economy:
- **Gold**: Primary currency
- **Wood**: Building and units
- **Stone**: Advanced buildings and upgrades
- **Food**: Unit training
- **Mana**: Magic units and research

### 7. UI Components

#### Research Panel (`src/ui/ResearchPanel.tsx`)
- Category-based research tree display
- Real-time research progress tracking
- Prerequisite visualization
- Cost and effect display
- Research queue system
- Completion statistics

#### Building Panel (`src/ui/BuildingPanel.tsx`)
- All building types and their upgrades
- Cost escalation display
- Unlock requirements
- Effect descriptions per level
- Build and upgrade costs

#### Unit Upgrade Panel (`src/ui/UnitUpgradePanel.tsx`)
- All unit types and stats
- Training costs
- Available upgrades per unit
- Stat bonus visualization
- Ability descriptions
- Unlock requirements

## Resource Sink Summary

### Early Game (Starting - 2000 gold spent)
- Basic military research (400-600 gold)
- Basic economy research (300-500 gold)
- Town Center Level 2 (800 gold)
- Barracks upgrades (400 gold)
- Basic unit training (100-200 gold per unit)

### Mid Game (2000-10000 gold spent)
- Advanced military research (800-1200 gold)
- Building unlocks (500-1000 gold)
- Town Center Level 3-4 (1200-2000 gold)
- Multiple building upgrades (1500-3000 gold)
- Unit upgrade research (600-1200 gold)
- Siege units unlock (500 gold)

### Late Game (10000+ gold spent)
- Elite military research (1200-2000 gold)
- Magic research tree (1200-2400 gold)
- Hero units (1000 gold unlock + 500 per unit)
- Town Center Level 5 (2500+ gold)
- Academy research (1500+ gold)
- Multiple unit upgrades (2000-4000 gold)
- Advanced abilities (1200+ gold)
- Temple and mana systems (800+ gold)

### Total Resource Sink Potential
- **Research alone**: ~20,000+ gold across all technologies
- **Building upgrades**: ~15,000+ gold for full building progression
- **Unit upgrades**: ~10,000+ gold for full unit upgrade tree
- **Total**: 45,000+ gold worth of late-game content

## Integration Example

```typescript
import { ResearchStore } from './store/researchStore';
import { GameStore } from './store/gameStore';
import { ResearchPanel } from './ui/ResearchPanel';

// Initialize stores
const researchStore = new ResearchStore();
const gameStore = new GameStore(researchStore);

// Game loop
function gameLoop(deltaTime: number) {
  // Update research progress
  const completedResearch = researchStore.updateResearch(deltaTime);
  
  if (completedResearch) {
    console.log('Research completed:', completedResearch.researchId);
    
    // Check for building/unit unlocks
    const research = researchStore.getResearch(completedResearch.researchId);
    research?.effects.forEach(effect => {
      if (effect.type === 'unlock_building') {
        gameStore.unlockBuilding(effect.target as any);
      }
      if (effect.type === 'unlock_unit') {
        gameStore.unlockUnit(effect.target as any);
      }
    });
  }
  
  // Update game state (buildings, resource generation)
  gameStore.update(deltaTime);
}

// Render UI
function renderUI() {
  return (
    <div>
      <ResearchPanel 
        researchStore={researchStore} 
        gameStore={gameStore} 
      />
    </div>
  );
}
```

## Key Design Decisions

1. **Progressive Unlock System**: Advanced features unlock through research, creating a sense of progression
2. **Escalating Costs**: Upgrade costs increase per level to maintain challenge
3. **Multiple Resource Types**: Diversified economy prevents one-dimensional gameplay
4. **Prerequisite Chains**: Creates meaningful tech progression and strategic choices
5. **Passive Benefits**: Buildings generate resources, encouraging investment
6. **Variety of Upgrades**: Military, economic, and magical paths for different playstyles

## Future Enhancements

Potential additions for even more late-game content:
- Wonder buildings (massive cost, unique benefits)
- Legendary hero upgrades (Level 3-5)
- Advanced magic spells requiring multiple research paths
- Alliance/Diplomacy systems
- Trade routes between buildings
- Artifact system (unique items requiring massive investment)
- Prestige system (reset with permanent bonuses)

## Files Modified/Created

✅ `src/types/research.ts` - Research system type definitions
✅ `src/types/buildings.ts` - Building system type definitions  
✅ `src/types/units.ts` - Unit system type definitions
✅ `src/store/researchStore.ts` - Research management and progression
✅ `src/store/gameStore.ts` - Game state, resources, buildings, and units
✅ `src/ui/ResearchPanel.tsx` - Research UI component
✅ `src/ui/BuildingPanel.tsx` - Building upgrade UI component
✅ `src/ui/UnitUpgradePanel.tsx` - Unit upgrade UI component
✅ `src/index.ts` - Main export file
✅ `src/README.md` - This documentation

## Testing

To test the system:

1. Initialize the stores:
```typescript
const researchStore = new ResearchStore();
const gameStore = new GameStore(researchStore);
```

2. Add resources for testing:
```typescript
gameStore.addResources({ gold: 10000, wood: 5000, stone: 5000, food: 5000, mana: 1000 });
```

3. Start research:
```typescript
const available = researchStore.getAvailableResearches();
researchStore.startResearch(available[0].id, gameStore.getResources());
```

4. Test building upgrades:
```typescript
const buildings = gameStore.getUnlockedBuildings();
// Implement building upgrade logic in your game
```

5. Test unit training:
```typescript
gameStore.trainUnit('warrior');
```

## Notes

- All costs and times can be adjusted for game balance
- The system is designed to be extensible - add more research, buildings, or units easily
- Resource generation rates should be tuned based on gameplay testing
- The UI components use Tailwind CSS for styling (ensure it's configured in your project)
- TypeScript types are fully defined for type safety

## Author

Created for Orca RTS to address Linear issue ORC-167: "Not enough things to spend resources on late game"
