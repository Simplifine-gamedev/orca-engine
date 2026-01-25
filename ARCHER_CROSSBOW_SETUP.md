# Archer & Crossbowman Unit Implementation Guide

## Quick Start

This document provides a quick reference for implementing archer and crossbowman units in the Orca RTS game.

## What's Been Created

### ✅ Configuration Files

1. **`/workspace/src/config/factions.ts`**
   - TypeScript type definitions and configuration
   - Defines archer, crossbowman, and archery range
   - Includes helper functions for unit/building lookup

2. **`/workspace/generated_factions/all_factions_characters.json`**
   - Complete character and building data
   - Stats, costs, animations, projectiles
   - Upgrade system definitions
   - 3D model generation prompts

### ✅ GDScript Implementation

3. **`/workspace/src/units/ranged_unit.gd`**
   - Base class for all ranged units
   - Movement, targeting, and attack logic
   - Projectile firing system
   - Health and damage handling

4. **`/workspace/src/units/archer.gd`**
   - Archer-specific implementation
   - Extends RangedUnit with archer stats

5. **`/workspace/src/units/crossbowman.gd`**
   - Crossbowman-specific implementation
   - Includes armor bonus (20% damage reduction)

6. **`/workspace/src/units/projectile.gd`**
   - Generic projectile system
   - Arc-based trajectory
   - Impact detection and damage

7. **`/workspace/src/buildings/archery_range.gd`**
   - Training facility for ranged units
   - Training queue system
   - Rally point support
   - Resource management integration points

### ✅ Tooling

8. **`/workspace/src/generate_unit_models.py`**
   - Automated 3D model generation
   - Uses Orca backend 3D API
   - Generates all required models from prompts

9. **`/workspace/src/README.md`**
   - Comprehensive documentation
   - Scene setup instructions
   - Usage examples and integration guide

## Implementation Checklist

### Phase 1: 3D Assets ⏳

- [ ] **Option A: Auto-generate models**
  ```bash
  cd /workspace/src
  python3 generate_unit_models.py
  ```

- [ ] **Option B: Create models manually**
  - Archer model (low poly, ~1000 tris)
  - Crossbowman model (low poly, ~1200 tris)
  - Archery Range building (low poly, ~2000 tris)
  - Arrow projectile (simple, ~100 tris)
  - Bolt projectile (simple, ~100 tris)

- [ ] **Add animations to unit models**
  - idle (looping, 2s)
  - walk (looping, 1s)
  - shoot_bow / shoot_crossbow (non-looping, 1.25-2s)
  - death (non-looping, 2s)

### Phase 2: Godot Scenes ⏳

- [ ] **Create projectile scenes**
  - `res://scenes/projectiles/arrow.tscn`
  - `res://scenes/projectiles/bolt.tscn`

- [ ] **Create unit scenes**
  - `res://scenes/units/archer.tscn`
    - Attach archer.gd script
    - Add model and AnimationPlayer
    - Add ProjectileSpawnPoint node
    - Configure projectile_scene export
  - `res://scenes/units/crossbowman.tscn`
    - Same structure as archer

- [ ] **Create building scene**
  - `res://scenes/buildings/archery_range.tscn`
    - Attach archery_range.gd script
    - Add model
    - Add UnitSpawnPoint and RallyPoint nodes
    - Configure unit scene exports

### Phase 3: Integration ⏳

- [ ] **Connect to Resource Manager**
  - Edit `archery_range.gd`:
    - `_has_resources()`
    - `_deduct_resources()`
    - `_refund_resources()`

- [ ] **Add to Faction System**
  - Register units in faction manager
  - Add to unit selection system

- [ ] **Create Training UI**
  - Building info panel
  - Unit training buttons (with costs)
  - Training queue display
  - Progress bar

### Phase 4: Polish ⏳

- [ ] **Visual Effects**
  - Projectile trail particles
  - Impact effects
  - Dust particles for movement

- [ ] **Audio**
  - Bow/crossbow shooting sounds
  - Arrow/bolt flying sounds
  - Impact sounds
  - Unit selection sounds
  - Death sounds

- [ ] **Unit Upgrades**
  - Implement upgrade system
  - Create upgrade UI
  - Apply stat modifications

## Quick Reference

### Unit Stats Comparison

| Stat | Archer | Crossbowman |
|------|--------|-------------|
| Health | 50 | 60 |
| Attack | 8 | 12 |
| Range | 7 | 8 |
| Speed | 3.5 | 3.0 |
| Attack Rate | 0.8/s | 0.5/s |
| DPS | 6.4 | 6.0 |
| Cost (Gold) | 40 | 60 |
| Cost (Wood) | 25 | 40 |
| Training Time | 20s | 30s |

**Archer**: Fast, mobile, good DPS  
**Crossbowman**: Tankier, longer range, armor piercing, slower

### File Paths Quick Reference

```
Configuration:
  src/config/factions.ts
  generated_factions/all_factions_characters.json

Scripts:
  src/units/ranged_unit.gd (base class)
  src/units/archer.gd
  src/units/crossbowman.gd
  src/units/projectile.gd
  src/buildings/archery_range.gd

Models (to be created):
  models/units/archer.glb
  models/units/crossbowman.glb
  models/buildings/archery_range.glb
  models/projectiles/arrow.glb
  models/projectiles/bolt.glb

Scenes (to be created):
  scenes/units/archer.tscn
  scenes/units/crossbowman.tscn
  scenes/buildings/archery_range.tscn
  scenes/projectiles/arrow.tscn
  scenes/projectiles/bolt.tscn
```

### Model Generation Prompts

**Archer**:
> "A medieval archer with a wooden longbow, wearing leather armor and a quiver of arrows on the back. Low poly game model style, suitable for RTS game."

**Crossbowman**:
> "A medieval crossbowman with a heavy crossbow, wearing chainmail armor and a helmet. Low poly game model style, suitable for RTS game. More armored than a regular archer."

**Archery Range**:
> "A medieval archery range building with training targets, weapon racks with bows and crossbows visible. Low poly game model style for RTS game. Building footprint approximately 3x3 units."

**Arrow**:
> "A simple medieval arrow projectile with wooden shaft and metal tip. Low poly style for RTS game."

**Bolt**:
> "A crossbow bolt projectile, thicker and shorter than an arrow with a heavier metal tip. Low poly style for RTS game."

## Testing

### Create a test scene to verify:

```gdscript
extends Node3D

@onready var archery_range = $ArcheryRange

func _ready():
    # Test training
    archery_range.train_unit("archer")
    archery_range.train_unit("crossbowman")
    
    # Set rally point
    archery_range.set_rally_point(Vector3(10, 0, 0))

func _process(_delta):
    # Monitor training
    if archery_range.is_training:
        print("Training progress: ", archery_range.get_training_progress() * 100, "%")
```

### Test unit behavior:

```gdscript
extends Node3D

@onready var archer = $Archer
@onready var target_dummy = $TargetDummy

func _ready():
    # Test movement
    await get_tree().create_timer(1.0).timeout
    archer.move_to(Vector3(5, 0, 0))
    
    # Test attacking
    await get_tree().create_timer(3.0).timeout
    archer.set_target(target_dummy)
```

## Next Features to Implement

After completing the basic archer/crossbowman implementation:

1. **Melee Units** - Swordsman, Spearman
2. **Cavalry Units** - Knight, Horse Archer
3. **Siege Units** - Catapult, Ballista
4. **Unit Formations** - Line, Box, Wedge
5. **Group Commands** - Select and control multiple units
6. **Unit Experience** - Veterancy system
7. **Special Abilities** - Volley fire, shield wall, etc.

## Troubleshooting

**Problem**: Models not showing  
**Solution**: Check model paths in inspector, ensure .glb files are in correct location

**Problem**: Projectiles not firing  
**Solution**: Verify projectile_scene and projectile_spawn_point are set in unit inspector

**Problem**: Units not moving  
**Solution**: Ensure CharacterBody3D is used and move_and_slide() is working

**Problem**: Training not starting  
**Solution**: Check that unit scenes are assigned in Archery Range inspector

## Architecture Notes

### Why TypeScript + JSON + GDScript?

- **TypeScript** (`factions.ts`): Type-safe configuration for external tools, editors, or web interfaces
- **JSON** (`all_factions_characters.json`): Runtime data loaded by game, easily editable
- **GDScript**: Game logic and behavior in Godot engine

This separation allows:
- External tools to use TypeScript types
- Game balance tweaks via JSON editing
- Performance-critical logic in GDScript
- 3D model generation from JSON prompts

## Resources

- **Documentation**: See `/workspace/src/README.md` for detailed guide
- **Configuration**: Edit `/workspace/generated_factions/all_factions_characters.json` for stats
- **Code**: All scripts in `/workspace/src/` with comments and documentation
- **3D Generation**: Use `/workspace/src/generate_unit_models.py` for automated asset creation

---

**Status**: Configuration and scripts complete ✅  
**Next**: Create 3D assets and Godot scenes ⏳  
**Then**: Integration and testing ⏳
