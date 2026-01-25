# RTS Visual Improvements Guide

**Issue**: ORC-142 - Game looks dark, units hard to see

## Problem Summary

Users reported that RTS games built with Orca Engine appear too dark, with units that are hard to see and feel like they're floating in dark space. This guide provides solutions to improve visual clarity and game visibility.

## Solutions Implemented

### 1. Enhanced Lighting System (`environment_setup.gd`)

The improved lighting system addresses darkness issues through:

#### Key Improvements:
- **Increased Ambient Light**: Energy increased from 1.0 to 1.5 for better base visibility
- **Brighter Directional Light**: Sun energy set to 1.8 for stronger illumination
- **Optimized Shadow Opacity**: Reduced to 0.6 to prevent excessive darkening
- **Better Sky Configuration**: Procedural sky with properly tuned horizon colors
- **Strategic Fog**: Light fog adds depth without obscuring units
- **Subtle Glow Effect**: Helps units stand out against the background
- **Reduced SSAO Impact**: Ambient occlusion light affect reduced to 0.3

#### Usage:
```gdscript
# Add WorldEnvironment node with environment_setup.gd script
# Add DirectionalLight3D as sibling node
# Adjust exported parameters in the inspector
```

#### Key Parameters:
- `ambient_light_energy`: 1.5 (increase for brighter scenes)
- `sun_energy`: 1.8 (increase for more dramatic lighting)
- `shadow_opacity`: 0.6 (lower = lighter shadows)
- `sun_angle_degrees`: -45° (adjust for time of day)

### 2. Improved Terrain Visibility (`terrain_visibility.gd`)

Creates terrain that provides good contrast with units:

#### Features:
- **Optimized Base Colors**: Grass green with good contrast
- **Subtle Emission**: Terrain emits slight light (0.1 * base_color)
- **Triplanar Mapping**: Better texturing on varied terrain
- **Optional Grid Overlay**: RTS-style grid for gameplay clarity
- **Height-Based Coloring**: Visual variety through altitude-based color variation
- **Procedural Terrain Generation**: Built-in height map generation with noise

#### Usage:
```gdscript
# Attach to MeshInstance3D node
# Call create_terrain_mesh(100, 100, 1.0) to generate terrain
# Or setup_terrain_material() to improve existing terrain
```

#### Key Features:
- Grid overlay option for RTS-style visibility
- Height-based color variation for visual interest
- Proper normal calculation for realistic lighting
- Adjustable roughness and base colors

### 3. Vegetation System (`vegetation_system.gd`)

Adds environmental objects to break up empty space:

#### Improvements:
- **Procedural Trees**: Simple but effective tree generation
- **Rock Placement**: Natural-looking rock distribution
- **Density Controls**: Adjustable density for performance/aesthetics
- **Smart Placement**: Avoids steep slopes and maintains spacing
- **Brightness Adjustment**: Vegetation slightly brighter (1.2x) for visibility
- **Shadow Casting**: Optional shadows for depth

#### Usage:
```gdscript
# Add as Node3D in scene
# Set terrain reference
# Adjust density parameters
# Call populate_vegetation() or it runs automatically
```

#### Key Parameters:
- `tree_density`: 0.15 (15% coverage)
- `rock_density`: 0.08 (8% coverage)
- `min_distance_between_objects`: 3.0 units
- `vegetation_brightness`: 1.2 (20% brighter)

### 4. Unit Material System (`unit_material_improved.gd`)

Makes units highly visible and distinguishable:

#### Features:
- **High-Contrast Team Colors**: Pre-defined vibrant color palette
- **Emission Effect**: Units glow slightly (30% emission energy)
- **Outline System**: Optional white outline for definition
- **Selection Indication**: Pulsing yellow highlight when selected
- **Rim Lighting**: Edge highlights for better definition
- **Optimized Materials**: Balance between visibility and performance

#### Usage:
```gdscript
# Attach to unit MeshInstance3D
# Set team_color to one of 8 predefined colors
# Enable emission and outline for maximum visibility
# Call set_selected(true) to highlight selected units
```

#### Team Colors (All High Visibility):
1. Red: `Color(0.9, 0.2, 0.2)`
2. Blue: `Color(0.2, 0.4, 0.9)`
3. Green: `Color(0.2, 0.8, 0.3)`
4. Yellow: `Color(0.9, 0.8, 0.2)`
5. Purple: `Color(0.8, 0.2, 0.8)`
6. Cyan: `Color(0.2, 0.8, 0.8)`
7. Orange: `Color(0.9, 0.5, 0.2)`
8. White: `Color(0.9, 0.9, 0.9)`

## Complete Scene Setup

The `improved_rts_scene.tscn` file demonstrates all improvements together:

```
ImprovedRTSScene (Node3D)
├── WorldEnvironment (with environment_setup.gd)
├── DirectionalLight3D (configured for RTS)
├── Terrain (MeshInstance3D with terrain_visibility.gd)
├── VegetationSystem (Node3D with vegetation_system.gd)
├── Camera3D (orthographic, angled for RTS view)
└── ReferenceUnit (MeshInstance3D with unit_material_improved.gd)
```

## Implementation Checklist

For existing RTS projects, follow these steps:

### Step 1: Lighting (Highest Priority)
- [ ] Add or modify WorldEnvironment node
- [ ] Attach `environment_setup.gd` script
- [ ] Increase `ambient_light_energy` to 1.5+
- [ ] Set `sun_energy` to 1.8+
- [ ] Reduce `shadow_opacity` to 0.6

### Step 2: Unit Visibility
- [ ] Apply improved materials to all units
- [ ] Use `unit_material_improved.gd` or similar approach
- [ ] Enable emission on unit materials
- [ ] Add outline effect for definition
- [ ] Use high-contrast team colors

### Step 3: Terrain
- [ ] Improve terrain material brightness
- [ ] Add height-based color variation
- [ ] Consider grid overlay for gameplay
- [ ] Ensure terrain doesn't absorb too much light

### Step 4: Environment
- [ ] Add vegetation system
- [ ] Place trees and rocks for visual interest
- [ ] Adjust density based on performance
- [ ] Ensure vegetation is slightly brighter

### Step 5: Camera & Post-Processing
- [ ] Use orthographic camera for RTS
- [ ] Enable glow/bloom (subtle)
- [ ] Adjust exposure if needed
- [ ] Test in various lighting conditions

## Performance Considerations

### Optimization Tips:
1. **LOD System**: Use Level of Detail for vegetation when many objects present
2. **Shadow Distance**: Limit `directional_shadow_max_distance` to 200 units
3. **Shadow Splits**: Use 2 splits for low-end, 4 for high-end hardware
4. **Vegetation Density**: Start low and increase based on target hardware
5. **Glow/SSAO**: Disable on low-end hardware if needed

### Performance Budget:
- **High-End**: All features enabled, 4 shadow splits, max vegetation
- **Mid-Range**: 2 shadow splits, medium vegetation, SSAO enabled
- **Low-End**: Minimal shadows, low vegetation, disable SSAO/glow

## Before and After Comparison

### Before (Common Issues):
- Ambient light: 1.0 (too dark)
- Sun energy: 1.0 (not bright enough)
- Shadow opacity: 1.0 (too harsh)
- Empty terrain (floating feeling)
- Dark unit colors (hard to see)
- No emission effects

### After (Improvements):
- Ambient light: 1.5 (50% brighter)
- Sun energy: 1.8 (80% brighter)
- Shadow opacity: 0.6 (40% lighter)
- Vegetation and environmental objects
- Bright, high-contrast unit colors
- Subtle emission and glow effects

## Testing Checklist

Test your improved scene:

- [ ] Units are clearly visible from typical gameplay camera
- [ ] Team colors are distinct and recognizable
- [ ] Terrain provides good contrast with units
- [ ] Shadows add depth without excessive darkening
- [ ] Environment feels populated, not empty
- [ ] Selected units have clear visual feedback
- [ ] Performance is acceptable on target hardware
- [ ] Scene looks good in different lighting conditions

## Additional Resources

### Godot Documentation:
- [Environment and Post-Processing](https://docs.godotengine.org/en/stable/tutorials/3d/environment_and_post_processing.html)
- [Lights and Shadows](https://docs.godotengine.org/en/stable/tutorials/3d/lights_and_shadows.html)
- [Standard Material 3D](https://docs.godotengine.org/en/stable/classes/class_standardmaterial3d.html)

### Best Practices:
1. **Test Early**: Check visibility in prototype stages
2. **Player Feedback**: Ask if units/terrain are clearly visible
3. **Multiple Monitors**: Test on different screens/brightness
4. **Accessibility**: Ensure colorblind-friendly team colors
5. **Iterate**: Continuously tune lighting based on gameplay

## Troubleshooting

### Units Still Too Dark?
- Increase `ambient_light_energy` further (try 2.0)
- Increase unit `emission_energy_multiplier` (try 0.5)
- Reduce `shadow_opacity` more (try 0.4)
- Check if unit materials are too dark

### Performance Issues?
- Reduce vegetation density
- Lower shadow quality (use 2 splits)
- Disable SSAO or reduce intensity
- Limit shadow distance
- Use LOD for distant objects

### Terrain Too Bright?
- Reduce terrain emission
- Lower `ambient_light_energy` slightly
- Adjust `base_color` to darker shade
- Increase terrain roughness

### No Depth Perception?
- Add fog (subtle)
- Ensure shadows are enabled
- Add height variation to terrain
- Place vegetation for scale reference
- Use SSAO for contact shadows

## Version History

- **v1.0** (2026-01-25): Initial visual improvement package
  - Enhanced lighting system
  - Improved terrain visibility
  - Vegetation system
  - Unit material improvements

## License

These examples are part of Orca Engine and follow the same licensing terms. See main LICENSE file for details.

## Contact

For issues or improvements, please file a bug report or feature request in the Orca Engine repository.
