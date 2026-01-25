# Quick Start Guide - RTS Map Decorations Demo

## Overview

This demo showcases environmental decorations for RTS maps, solving the "plain map" problem described in Linear issue ORC-143.

## What You'll See

When you run this demo, you'll see a procedurally generated map with:

- **Terrain**: Hills, valleys, and multiple biomes
- **Vegetation**: Thousands of grass patches, bushes, flowers, and mushrooms
- **Trees**: Pine, oak, and birch trees scattered across the landscape
- **Rocks**: Small, medium, and large rocks plus massive boulders
- **Dead Trees**: For atmospheric variety

## Running the Demo

### Step 1: Open in Orca Engine

```bash
# If you have Orca Engine built:
cd /path/to/orca-engine
./bin/godot.*.editor.* demos/rts-demo/project.godot

# Or use the editor to open:
# File -> Open Project -> Navigate to demos/rts-demo/
```

### Step 2: Run the Main Scene

1. The editor will open the project
2. Press **F5** or click the **Play** button
3. The demo will generate and display the decorated map

## Interactive Controls

Once running, use these controls:

### Regeneration
- **R** - Regenerate terrain only
- **V** - Regenerate vegetation only  
- **D** - Regenerate decorations (rocks/trees) only
- **A** - Regenerate everything

### Camera
- **Left/Right Arrow** - Rotate camera around map
- **Page Up/Down** - Zoom in/out

## Customization

### In the Editor

Select nodes in the scene tree to adjust parameters:

#### HeightmapTerrain Node
- `terrain_size`: Map dimensions (default: 100x100)
- `height_scale`: Maximum terrain height (default: 10)
- `resolution`: Terrain detail level (default: 100)
- `noise_frequency`: Controls terrain feature size
- `enable_hills`: Toggle hills
- `enable_valleys`: Toggle valleys

#### VegetationSystem Node
- `vegetation_density`: Base density (default: 0.5)
- `grass_density`: Grass multiplier (default: 2.0)
- `bush_density`: Bush multiplier (default: 0.3)
- `flower_density`: Flower multiplier (default: 0.5)
- `mushroom_density`: Mushroom multiplier (default: 0.2)
- `tall_grass_density`: Tall grass multiplier (default: 1.0)

#### DecorationSpawner Node
- `rock_density`: Rocks per 100m² (default: 5.0)
- `tree_density`: Trees per 100m² (default: 3.0)
- `boulder_density`: Boulders per 100m² (default: 1.0)
- `dead_tree_density`: Dead trees per 100m² (default: 0.5)
- `min_separation_distance`: Space between large objects (default: 3.0)

### Performance Tuning

For slower computers, reduce these values:
- VegetationSystem: Lower all density values by 50%
- DecorationSpawner: Reduce density values by 30-50%
- HeightmapTerrain: Lower `resolution` to 75 or 50

For more detail:
- Increase all density values
- Increase terrain resolution to 150+
- Reduce `min_separation_distance` for denser forests

## Expected Performance

On a modern computer (2020+):
- **Terrain**: Instant generation
- **Vegetation**: 1-2 seconds for ~5,000 plants
- **Decorations**: < 1 second for ~300 objects
- **Runtime**: 60+ FPS

## Troubleshooting

### "Script error" or "Parse error"
- Ensure you're using Godot 4.x (Orca Engine is based on Godot 4)
- Check that all script paths are correct

### Poor performance
- Reduce density settings in the Inspector
- Lower terrain resolution
- Check GPU drivers are up to date

### Map looks empty
- Check that enable flags are true for each system
- Verify density values are > 0
- Try pressing 'A' to regenerate all systems

### Camera stuck or controls not working
- Make sure the game window has focus
- Check console for any error messages
- Try restarting the demo

## Understanding the Code

### Key Classes

**VegetationSystem** (`scripts/vegetation_system.gd`)
- Main function: `generate_vegetation()`
- Uses MultiMesh for performance
- Samples terrain height for placement

**DecorationSpawner** (`scripts/decoration_spawner.gd`)
- Main function: `generate_decorations()`
- Implements collision avoidance
- Creates individual mesh instances

**HeightmapTerrain** (`scripts/heightmap_terrain.gd`)
- Main function: `generate_terrain()`
- Uses FastNoiseLite for procedural generation
- Provides `get_height_at()` for other systems

**DemoController** (`scripts/demo_controller.gd`)
- Handles keyboard input
- Coordinates regeneration
- Manages camera movement

### Extending the Demo

Want to add more features? Try:

1. **Custom vegetation types**: Add new enum values and mesh creation functions
2. **More terrain biomes**: Extend `_get_terrain_color()` in HeightmapTerrain
3. **Animated vegetation**: Add wind shaders to grass and trees
4. **Seasonal changes**: Create variants with different colors
5. **Interactive elements**: Make trees destructible or harvestable

## Integration into Your RTS Game

To use this in your own project:

1. Copy the three main scripts to your project
2. Create nodes in your map scene:
   ```
   MapRoot (Node3D)
   ├── Terrain (Node3D) + heightmap_terrain.gd
   ├── Vegetation (Node3D) + vegetation_system.gd
   └── Decorations (Node3D) + decoration_spawner.gd
   ```
3. Set node path references in the Inspector:
   - VegetationSystem.terrain → Terrain node
   - DecorationSpawner.terrain → Terrain node
4. Adjust map_size to match your game map
5. Replace procedural meshes with your art assets
6. Add exclusion zones around bases/resources

## Next Steps

- Read `IMPLEMENTATION_NOTES.md` for technical details
- Check `LINEAR_ISSUE_RESOLUTION.md` for context
- Explore the code to understand the systems
- Experiment with parameters to find your ideal look
- Replace placeholder meshes with proper 3D models

## Support

For questions or issues:
- Check the code comments (heavily documented)
- Review the main README.md
- Open an issue on the Orca Engine repository

## Have Fun!

Experiment with different settings, try extreme values, and see what interesting landscapes you can create. The procedural nature means every regeneration creates a unique map!

**Tip**: Try setting `rock_density = 20` and `tree_density = 15` for a dense forest environment, or `height_scale = 30` with `noise_frequency = 0.02` for dramatic mountain ranges!
