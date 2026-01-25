# RTS Demo - Map Visual Decorations

This demo showcases environmental decorations for an RTS game map, including:

- Vegetation system (grass, bushes, flowers)
- Rock scatter system (various sizes)
- Tree placement system
- Terrain heightmap support
- Procedural decoration placement

## Features

### VegetationSystem
- Procedurally places vegetation across the terrain
- Supports multiple vegetation types (grass, bushes, flowers, mushrooms)
- Density-based placement with randomization
- Performance-optimized with MultiMesh

### DecorationSpawner
- Scatter rocks of various sizes
- Place trees with variety
- Support for custom decoration placement rules
- Avoid overlap with game objects

### HeightmapTerrain
- Basic terrain with heightmap support
- Biome-based decoration placement
- Terrain feature support (hills, cliffs)

## Usage

Open this project in Orca Engine and run the main scene to see the decorated map.

## Files

- `main.tscn` - Main scene with terrain and decorations
- `scripts/vegetation_system.gd` - Vegetation placement logic
- `scripts/decoration_spawner.gd` - Rock and tree placement
- `scripts/heightmap_terrain.gd` - Terrain management
- `assets/` - Placeholder meshes and textures
