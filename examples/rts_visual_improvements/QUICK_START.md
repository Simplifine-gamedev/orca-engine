# Quick Start: Fixing Dark RTS Game

**Issue: ORC-142** - Units hard to see, game looks dark

## 3-Minute Fix

### 1. Lighting (Most Important) ⚡

Open your main scene and adjust the WorldEnvironment:

```gdscript
# In your WorldEnvironment script or inspector:
environment.ambient_light_energy = 1.5  # Was probably 1.0
environment.ambient_light_color = Color(0.8, 0.85, 0.9)
```

Adjust your DirectionalLight3D (sun):

```gdscript
# In your DirectionalLight3D:
light_energy = 1.8  # Was probably 1.0
light_color = Color(1.0, 0.95, 0.85)
shadow_opacity = 0.6  # Was probably 1.0
```

**Result**: 50-80% brighter scene immediately!

### 2. Unit Visibility 🎯

Add emission to your unit materials:

```gdscript
# For each unit's material:
material.emission_enabled = true
material.emission = team_color  # Your unit's team color
material.emission_energy_multiplier = 0.3
```

**Result**: Units glow slightly and are much easier to see!

### 3. Terrain (Optional but Recommended) 🌍

If terrain is too dark:

```gdscript
# In your terrain material:
material.emission_enabled = true
material.emission = terrain_color * 0.1  # Very subtle glow
material.emission_energy_multiplier = 0.2
```

**Result**: Terrain has better base visibility!

## Copy-Paste Solutions

### WorldEnvironment Script (Minimal)

```gdscript
extends WorldEnvironment

func _ready():
    if not environment:
        environment = Environment.new()
    
    # Brighter ambient light
    environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    environment.ambient_light_color = Color(0.8, 0.85, 0.9)
    environment.ambient_light_energy = 1.5
    
    # Better tone mapping
    environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
    environment.tonemap_exposure = 1.1
```

### Unit Material Script (Minimal)

```gdscript
extends MeshInstance3D

@export var team_color := Color(0.9, 0.2, 0.2)

func _ready():
    var mat = StandardMaterial3D.new()
    mat.albedo_color = team_color * 1.2  # 20% brighter
    mat.emission_enabled = true
    mat.emission = team_color
    mat.emission_energy_multiplier = 0.3
    mat.roughness = 0.6
    material_override = mat
```

## Quick Test

After applying changes, ask yourself:
1. ✅ Can I clearly see units against the terrain?
2. ✅ Are team colors distinct and recognizable?
3. ✅ Does the scene feel bright enough?
4. ✅ Are shadows visible but not too dark?

If all answers are YES, you're done! ✨

## Need More?

See full `README.md` for:
- Vegetation system to fill empty space
- Advanced terrain setup
- Unit outline effects
- Complete scene example

## Values at a Glance

| Setting | Before | After | Change |
|---------|--------|-------|--------|
| Ambient Energy | 1.0 | 1.5 | +50% |
| Sun Energy | 1.0 | 1.8 | +80% |
| Shadow Opacity | 1.0 | 0.6 | -40% |
| Unit Emission | None | 0.3 | +30% |

These values should work for most RTS games. Adjust to taste!
