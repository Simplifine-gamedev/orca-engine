# RTS Units System - Archer & Crossbowman

This directory contains the implementation for ranged units (Archer and Crossbowman) in the Orca RTS game.

## Overview

The system consists of:

1. **Configuration Files** - TypeScript and JSON definitions for units, buildings, and projectiles
2. **GDScript Classes** - Godot implementation of unit behavior
3. **3D Model Generation** - Scripts to generate required 3D assets
4. **Building System** - Archery Range for training ranged units

## Directory Structure

```
/workspace/
├── src/
│   ├── config/
│   │   └── factions.ts              # TypeScript faction configuration
│   ├── units/
│   │   ├── ranged_unit.gd           # Base class for ranged units
│   │   ├── archer.gd                # Archer unit implementation
│   │   ├── crossbowman.gd           # Crossbowman unit implementation
│   │   └── projectile.gd            # Projectile system
│   ├── buildings/
│   │   └── archery_range.gd         # Archery Range building
│   ├── generate_unit_models.py      # 3D model generation script
│   └── README.md                    # This file
└── generated_factions/
    └── all_factions_characters.json # Complete character data
```

## Unit Specifications

### Archer

**Role**: Fast, mobile ranged unit with moderate damage

**Stats**:
- Health: 50
- Attack: 8
- Attack Range: 7
- Move Speed: 3.5
- Attack Speed: 0.8 attacks/sec
- Cost: 40 gold, 25 wood, 1 food
- Training Time: 20 seconds

**Projectile**: Arrow with moderate arc (0.3), speed 15

**Animations Required**:
- `idle` - Standing idle
- `walk` - Moving
- `shoot_bow` - Shooting animation
- `death` - Death animation

### Crossbowman

**Role**: Heavily armored ranged unit with high damage but slower rate of fire

**Stats**:
- Health: 60
- Attack: 12
- Attack Range: 8
- Move Speed: 3.0
- Attack Speed: 0.5 attacks/sec
- Defense: 20% damage reduction
- Cost: 60 gold, 40 wood, 1 food
- Training Time: 30 seconds

**Projectile**: Crossbow bolt with flat arc (0.15), speed 20

**Animations Required**:
- `idle` - Standing idle
- `walk` - Moving
- `shoot_crossbow` - Shooting animation (includes reload)
- `death` - Death animation

## Archery Range

**Building Stats**:
- Health: 800
- Defense: 5
- Cost: 150 gold, 100 wood, 50 stone
- Build Time: 45 seconds

**Produces**: Archers and Crossbowmen

**Features**:
- Training queue system
- Rally point for spawned units
- Unit-specific training times

## Setting Up 3D Models

### Option 1: Generate Models Using Backend API

If you have the Orca backend running with 3D generation enabled:

```bash
cd /workspace/src
python3 generate_unit_models.py
```

This will generate:
- Archer model (`models/units/archer.glb`)
- Crossbowman model (`models/units/crossbowman.glb`)
- Archery Range model (`models/buildings/archery_range.glb`)
- Arrow projectile (`models/projectiles/arrow.glb`)
- Bolt projectile (`models/projectiles/bolt.glb`)

### Option 2: Manual Model Creation

Create models in Blender or your preferred 3D software following these guidelines:

**Unit Models**:
- Low poly style (500-1500 triangles)
- Scale: approximately 1.8 units tall
- Forward direction: -Z axis
- Include armature for animations
- Export as `.glb` format

**Projectile Models**:
- Very simple (50-200 triangles)
- Arrow: ~0.8 units long
- Bolt: ~0.6 units long, thicker than arrow
- Forward direction: +X axis

**Building Model**:
- Footprint: approximately 3x3 units
- Height: ~2.5 units
- Should visually indicate purpose (targets, weapon racks)

### Model Placement

Place generated or created models in:
```
/workspace/models/
├── units/
│   ├── archer.glb
│   └── crossbowman.glb
├── buildings/
│   └── archery_range.glb
└── projectiles/
    ├── arrow.glb
    └── bolt.glb
```

## Creating Godot Scenes

### 1. Archer Scene (`res://scenes/units/archer.tscn`)

```
Archer (Node3D) - Script: res://src/units/archer.gd
├── Model (Node3D) - Imported from archer.glb
│   └── AnimationPlayer
├── ProjectileSpawnPoint (Node3D) - Position: (0, 1.5, 0.5)
└── CollisionShape3D - For selection/collision
```

**Scene Setup**:
1. Create new 3D Scene
2. Rename root to "Archer"
3. Attach `archer.gd` script
4. Add imported archer model as child
5. Add Node3D as "ProjectileSpawnPoint" (position slightly in front and up)
6. Configure exports in inspector:
   - Assign AnimationPlayer
   - Assign ProjectileSpawnPoint
   - Set projectile_scene to arrow scene

### 2. Crossbowman Scene (`res://scenes/units/crossbowman.tscn`)

Similar structure to Archer, but:
- Use crossbowman.glb model
- Use crossbowman.gd script
- Set projectile_scene to bolt scene

### 3. Projectile Scenes

**Arrow** (`res://scenes/projectiles/arrow.tscn`):
```
Arrow (Node3D) - Script: res://src/units/projectile.gd
├── Model (MeshInstance3D) - arrow.glb
└── (Optional) Trail particles
```

**Bolt** (`res://scenes/projectiles/bolt.tscn`):
```
Bolt (Node3D) - Script: res://src/units/projectile.gd
├── Model (MeshInstance3D) - bolt.glb
└── (Optional) Trail particles
```

### 4. Archery Range Scene (`res://scenes/buildings/archery_range.tscn`)

```
ArcheryRange (Node3D) - Script: res://src/buildings/archery_range.gd
├── Model (Node3D) - archery_range.glb
├── UnitSpawnPoint (Node3D) - Position in front of building
└── RallyPoint (Node3D) - Default rally position
```

**Inspector Configuration**:
- Set `archer_scene` to archer.tscn
- Set `crossbowman_scene` to crossbowman.tscn
- Assign UnitSpawnPoint and RallyPoint nodes

## Usage in Game

### Training Units

```gdscript
var archery_range = $ArcheryRange

# Train an archer
archery_range.train_unit("archer")

# Train a crossbowman
archery_range.train_unit("crossbowman")

# Set rally point
archery_range.set_rally_point(Vector3(10, 0, 10))

# Check training progress
var progress = archery_range.get_training_progress()
print("Training: ", progress * 100, "%")
```

### Controlling Units

```gdscript
var archer = $Archer

# Move to position
archer.move_to(Vector3(15, 0, 20))

# Attack target
var enemy = $Enemy
archer.set_target(enemy)

# Stop current action
archer.stop()

# Check health
var health_pct = archer.get_health_percentage()
```

### Handling Events

```gdscript
# Building signals
archery_range.unit_training_started.connect(_on_training_started)
archery_range.unit_training_completed.connect(_on_training_completed)

func _on_training_completed(unit_type: String, unit: Node3D):
    print("Trained: ", unit_type)
    # Add unit to your army/selection group

# Unit signals
archer.unit_died.connect(_on_unit_died)
archer.target_acquired.connect(_on_target_acquired)
archer.projectile_fired.connect(_on_projectile_fired)
```

## Animations

Each unit requires these animations in the model's AnimationPlayer:

### Required Animations

1. **idle**: Looping, ~2 seconds
   - Character standing alert with weapon

2. **walk**: Looping, ~1 second
   - Walking cycle

3. **shoot_bow** (Archer) / **shoot_crossbow** (Crossbowman): Non-looping
   - Archer: Draw bow → release → return to ready (~1.25s)
   - Crossbowman: Aim → shoot → reload (~2.0s)
   - Projectile releases at 60% through animation

4. **death**: Non-looping, ~2 seconds
   - Unit falling/dying

### Animation Events

The `attack` animation should have a signal/event at the projectile release point (see `projectileReleaseTime` in character JSON).

## Upgrades System

Upgrades defined in `all_factions_characters.json`:

**Archer Upgrades**:
- Fletching: +1 range
- Bodkin Arrows: +2 attack, +2 armor piercing
- Bracer: +0.2 attack speed

**Crossbowman Upgrades**:
- Steel Bolts: +3 attack
- Pavise: +3 defense

These can be researched at the Archery Range (requires UI implementation).

## Integration with Game Systems

### Resource Manager

The building system expects a resource manager with these methods:
- `has_resources(cost: Dictionary) -> bool`
- `deduct_resources(cost: Dictionary)`
- `refund_resources(amount: Dictionary)`

Connect these in `archery_range.gd` methods:
- `_has_resources()`
- `_deduct_resources()`
- `_refund_resources()`

### Selection System

Units should be added to selection groups. They support standard RTS commands:
- Move to position
- Attack target
- Stop/Hold position

### UI Integration

Create UI to display:
- Unit training queue
- Training progress
- Unit costs
- Health bars
- Selection info

## Testing

### Quick Test Scene

Create a test scene to verify functionality:

```gdscript
extends Node3D

func _ready():
    # Spawn archery range
    var archery_range = preload("res://scenes/buildings/archery_range.tscn").instantiate()
    add_child(archery_range)
    archery_range.position = Vector3(0, 0, 0)
    
    # Train some units
    archery_range.train_unit("archer")
    archery_range.train_unit("crossbowman")
    
    # Set rally point
    archery_range.set_rally_point(Vector3(5, 0, 0))
```

## Next Steps

1. **Create 3D Models**: Either generate using the script or create manually
2. **Set Up Scenes**: Create Godot scenes as described above
3. **Add Animations**: Ensure all required animations are present
4. **Integrate Systems**: Connect to resource manager and selection system
5. **Create UI**: Build training interface and unit info display
6. **Add Effects**: Particle effects for projectile trails and impacts
7. **Audio**: Add sound effects for attacks, training, etc.
8. **Balance**: Playtest and adjust unit stats as needed

## Configuration Reference

All unit and building data is centralized in:
- `/workspace/src/config/factions.ts` - TypeScript type definitions
- `/workspace/generated_factions/all_factions_characters.json` - Game data

Edit these files to modify stats, costs, and other properties. The GDScript classes load configuration from the JSON file at runtime.

## Troubleshooting

**Units not attacking**:
- Check `projectile_scene` is assigned in inspector
- Verify `projectile_spawn_point` is set
- Ensure target has `take_damage()` method

**Animations not playing**:
- Verify AnimationPlayer is assigned
- Check animation names match exactly
- Ensure animations exist in the model

**Training not working**:
- Verify unit scenes are assigned in Archery Range
- Check resource manager integration
- Ensure UnitSpawnPoint is set

**Projectiles missing target**:
- Verify projectile speed and arc values
- Check target is valid when projectile is fired
- Ensure projectile script is attached

## Future Enhancements

- Formation system for grouped movement
- Unit veterancy and experience
- Special abilities (volley fire, aimed shot, etc.)
- Garrison system (units inside buildings)
- Different ammunition types
- Counter-unit bonuses (vs cavalry, vs armor, etc.)
