# Orca RTS Game - Archer & Crossbowman Units

This directory contains the implementation of archer and crossbowman units for the Orca RTS game, built on the Orca Engine (Godot-based).

## Overview

This implementation addresses issue ORC-162 by providing:

1. **Archer Unit** - Fast-firing ranged unit with bow
2. **Crossbowman Unit** - Powerful ranged unit with crossbow and armor penetration
3. **Ranged Attack Animations** - Attack, reload, and movement animations
4. **Projectile Visuals** - Arrow and crossbow bolt projectiles
5. **Archery Range Building** - Military building that trains both units

## File Structure

```
rts_game/
├── units/
│   ├── archer.gd              # Archer unit logic
│   ├── archer.tscn            # Archer scene
│   ├── crossbowman.gd         # Crossbowman unit logic
│   └── crossbowman.tscn       # Crossbowman scene
├── projectiles/
│   ├── arrow.gd               # Arrow projectile logic
│   ├── arrow.tscn             # Arrow scene
│   ├── crossbow_bolt.gd       # Crossbow bolt logic
│   └── crossbow_bolt.tscn     # Crossbow bolt scene
├── buildings/
│   ├── archery_range.gd       # Archery Range building logic
│   └── archery_range.tscn     # Archery Range scene
├── config/
│   └── factions.gd            # Faction configuration (GDScript)
└── assets/
    ├── models/                # 3D model files (to be added)
    ├── animations/            # Animation files (to be added)
    └── icons/                 # UI icons (to be added)
```

## Unit Specifications

### Archer

- **Health**: 60 HP
- **Damage**: 12
- **Range**: 15 units
- **Speed**: 3.5 units/sec
- **Attack Cooldown**: 1.5 seconds
- **Cost**: 50 wood, 25 gold
- **Training Time**: 30 seconds
- **Projectile**: Arrow (20 speed)

**Features**:
- Fast movement and attack speed
- Good for hit-and-run tactics
- Basic ranged infantry unit
- Effective against unarmored targets

### Crossbowman

- **Health**: 70 HP
- **Damage**: 18
- **Armor Penetration**: +5
- **Range**: 18 units
- **Speed**: 3.0 units/sec
- **Attack Cooldown**: 2.2 seconds
- **Cost**: 60 wood, 40 gold
- **Training Time**: 45 seconds
- **Projectile**: Crossbow Bolt (25 speed)

**Features**:
- Slower but more powerful than archer
- Armor penetration bonus
- Reload animation between attacks
- Effective against armored targets
- Longer range

## Archery Range Building

- **Health**: 500 HP
- **Cost**: 150 wood, 50 gold
- **Build Time**: 60 seconds
- **Trains**: Archer, Crossbowman

**Features**:
- Training queue system
- Training progress indicator
- Customizable rally point
- Unit spawning at designated spawn point

## Factions

Three factions are configured with archer and crossbowman units:

### 1. Human Kingdom
- Balanced units
- Standard stats
- Building: "Archery Range"

### 2. Orc Horde
- Higher health and damage
- Slightly slower
- Building: "War Lodge"

### 3. Elven Alliance
- Faster and longer range
- Lower health
- Building: "Hunter's Hall"

## Configuration Files

### GDScript Configuration
- `rts_game/config/factions.gd` - Main faction configuration for Godot

### TypeScript Configuration
- `src/config/factions.ts` - TypeScript interface for web integration

### JSON Data
- `generated_factions/all_factions_characters.json` - Complete character data

## Animation States

### Archer Animations
- `idle` - Standing with bow ready
- `walk` - Walking with bow
- `attack_bow` - Drawing and releasing arrow
- `death` - Death animation

### Crossbowman Animations
- `idle` - Standing with crossbow ready
- `walk` - Walking with crossbow
- `attack_crossbow` - Firing crossbow
- `reload` - Reloading crossbow
- `death` - Death animation

## Projectiles

### Arrow
- Visual: Brown wooden shaft with metal tip
- Speed: 20 units/sec
- Max Distance: 30 units
- Auto-destroys on impact or max distance

### Crossbow Bolt
- Visual: Dark wooden bolt with larger metal head
- Speed: 25 units/sec
- Max Distance: 35 units
- Heavier impact effect
- Applies armor penetration bonus

## Usage

### Spawning Units

```gdscript
# Load and spawn an archer
var archer_scene = preload("res://rts_game/units/archer.tscn")
var archer = archer_scene.instantiate()
add_child(archer)
archer.set_faction(0)  # Set to human faction
archer.set_target(enemy_unit)  # Set attack target
```

### Training from Archery Range

```gdscript
# Get archery range reference
var archery_range = $ArcheryRange

# Train an archer (if resources available)
archery_range.train_archer()

# Train a crossbowman
archery_range.train_crossbowman()

# Set rally point
archery_range.set_rally_point(Vector3(10, 0, 10))
```

### Using Faction Configuration

```gdscript
# Get unit data
var archer_data = FactionsConfig.get_unit_data("human", "archer")
print("Archer health: ", archer_data.health)
print("Archer damage: ", archer_data.damage)

# Get faction color
var faction_color = FactionsConfig.get_faction_color("human")
```

## Next Steps

### Required Assets

1. **3D Models**:
   - Archer character model with bow
   - Crossbowman character model with crossbow
   - Archery Range building model
   - Arrow projectile model (currently using procedural)
   - Crossbow bolt model (currently using procedural)

2. **Animations**:
   - Idle, walk, attack, death for archer
   - Idle, walk, attack, reload, death for crossbowman
   - Building construction/destruction animations

3. **UI Icons**:
   - Archer portrait icon
   - Crossbowman portrait icon
   - Archery Range icon
   - Faction-specific variants

4. **Sound Effects**:
   - Bow release sound
   - Crossbow fire sound
   - Arrow/bolt impact sounds
   - Unit selection/movement sounds

### Using 3D Generation

The Orca Engine backend supports 3D model generation. To generate models:

1. Configure the backend with 3D generation service
2. Use the editor's AI tools to generate models:
   - "Generate a 3D model of a medieval archer with a longbow"
   - "Generate a 3D model of a crossbowman in leather armor"

See `backend/3d-generation.md` for more details.

## Integration

### Godot Project Setup

1. Open the Orca Engine editor
2. Create or open your RTS game project
3. Copy the `rts_game` directory to your project
4. Instantiate units and buildings as needed
5. Configure faction data using `FactionsConfig`

### TypeScript Integration

For web-based tools or editors:

```typescript
import { getFactionData, getUnitData } from './src/config/factions';

const archerData = getUnitData('human', 'archer');
console.log(`Archer: ${archerData.health} HP, ${archerData.damage} DMG`);
```

## Testing

To test the units:

1. Create a test scene in Godot
2. Add an Archery Range building
3. Add terrain/ground
4. Add enemy targets
5. Train units and observe behavior
6. Test combat, movement, and animations

## License

Part of the Orca Engine project. See main LICENSE for details.

## Contributing

When adding new ranged units:
1. Extend the base unit scripts
2. Add faction-specific variants to configuration
3. Create proper 3D models and animations
4. Update `all_factions_characters.json`
5. Add appropriate icons and sound effects
