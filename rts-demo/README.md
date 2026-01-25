# Orca RTS - Environmental Decorations Demo

A demo RTS game showcasing environmental decorations including vegetation, rocks, trees, and dynamic terrain features.

## Features

### Vegetation System (`src/vegetation/VegetationSystem.tsx`)
- **Rocks**: Various sizes scattered across the terrain
- **Trees**: 
  - Pine trees (conical shapes)
  - Oak trees (rounded canopies)
  - Multiple varieties for visual diversity
- **Bushes**: Low vegetation scattered throughout
- **Grass Patches**: Ground-level vegetation
- **Flowers**: Colorful accents in various hues
- **Mushrooms**: Small decorative elements

### Terrain System (`src/terrain/HeightmapTerrain.tsx`)
- **Heightmap-based terrain**: Procedurally generated using Fractal Brownian Motion
- **Hills**: Rolling terrain with multiple octaves of noise
- **Cliffs**: Dramatic vertical features
- **Water**: Low-lying water plane with transparency
- **Procedural texturing**: Grass, dirt, and rocky areas based on terrain features

## Installation

```bash
cd rts-demo
npm install
```

## Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Building for Production

```bash
npm run build
npm run preview
```

## Controls

- **Mouse drag**: Orbit camera around the scene
- **Mouse wheel**: Zoom in/out
- **UI Controls**: Adjust vegetation density and toggle wireframe mode

## Technical Details

### Technologies Used
- **React 18**: UI framework
- **Three.js**: 3D rendering engine
- **React Three Fiber**: React renderer for Three.js
- **@react-three/drei**: Useful helpers and abstractions
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool

### Performance Optimizations
- **Instanced rendering**: All vegetation types use instanced meshes for efficient rendering
- **Procedural generation**: Seeded random generation ensures consistency
- **LOD-ready**: Architecture supports Level of Detail if needed
- **Efficient geometry**: Simple primitives for each decoration type

## Customization

### Adjusting Vegetation Density
Use the slider in the UI or modify the `density` prop in `GameScene.tsx`:

```tsx
<VegetationSystem
	terrainSize={100}
	density={1.0} // Adjust from 0.1 to 2.0
	seed={42}
/>
```

### Modifying Terrain
Adjust terrain parameters in `GameScene.tsx`:

```tsx
<HeightmapTerrain
	size={100} // Terrain size
	resolution={128} // Vertex resolution
	heightScale={10} // Maximum height
	seed={42} // Random seed
	showWireframe={false}
/>
```

### Adding New Vegetation Types
1. Add new type in `VegetationSystem.tsx` vegetation generation loop
2. Create new geometry for the type
3. Add to the instanced rendering section

## Architecture

```
rts-demo/
├── src/
│   ├── vegetation/
│   │   └── VegetationSystem.tsx    # All vegetation decorations
│   ├── terrain/
│   │   └── HeightmapTerrain.tsx    # Terrain generation & rendering
│   ├── GameScene.tsx                # Main 3D scene setup
│   ├── App.tsx                      # React app entry
│   └── main.tsx                     # DOM rendering
├── public/                          # Static assets
├── index.html                       # HTML template
├── package.json                     # Dependencies
├── vite.config.ts                   # Vite configuration
└── tsconfig.json                    # TypeScript configuration
```

## User Feedback Addressed

This implementation addresses the feedback from the Linear issue:

- ✅ **"Map looks so plain"**: Added 6 types of environmental decorations
- ✅ **"Vegetation"**: Grass patches, bushes, multiple tree varieties
- ✅ **"Rocks/trees around (decorations)"**: Scattered rocks of various sizes, pine and oak trees
- ✅ **More variety**: Flowers, mushrooms, different tree types
- ✅ **Terrain features**: Hills and cliffs for more interesting landscape

## Performance Notes

- Default density (1.0) spawns approximately 560 decoration objects
- All objects use instanced rendering for optimal GPU performance
- Terrain uses a 128x128 vertex grid (customizable)
- Shadows are enabled for realistic lighting

## Future Enhancements

Potential improvements:
- Add wind animation for trees and grass
- Implement LOD (Level of Detail) for distant objects
- Add seasonal variations (autumn colors, winter snow)
- Include more diverse rock and tree models
- Add particle effects (falling leaves, dust)
- Implement proper collision detection with terrain heightmap
- Add biome system (forest, desert, tundra areas)

## License

Part of the Orca Engine project.
