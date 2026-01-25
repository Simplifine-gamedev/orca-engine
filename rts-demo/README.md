# Orca RTS Demo

A visual demonstration of an RTS game built with React Three Fiber, addressing visual clarity issues.

## Features

### Lighting Improvements
- **Enhanced Ambient Light**: Increased ambient light intensity (0.8) with cool tint
- **Hemisphere Light**: Added for better ground-to-sky gradient lighting
- **Multiple Directional Lights**: Main sun light plus fill and rim lights
- **Color Temperature**: Warm sun (#fff5e6) with cool fill light (#b0d4ff)
- **Shadow Quality**: 2048x2048 shadow maps with proper bias

### Environment
- **Varied Terrain**: Heightmap with multiple noise octaves for natural hills
- **Vertex Colors**: Height-based coloring (gray rocks, green grass, brown dirt)
- **Grid Helper**: Subtle grid lines for depth perception
- **Sky System**: Procedural sky with sun position
- **Stars**: Subtle starfield for atmospheric depth

### Vegetation System
- **Trees**: 25 procedural trees with trunk and multi-layer crown
- **Rocks**: 40 varied rocks with different sizes and gray tones
- **Bushes**: 30 clustered sphere bushes
- **Grass Patches**: 50 detailed grass blade clusters

### Unit Visibility
- **Bright Colors**: High-saturation unit colors (red, blue, green, orange)
- **Emissive Materials**: Units emit light matching their team color
- **Outline Rings**: Transparent rings at unit base for identification
- **Point Lights**: Each unit has a colored point light
- **Shadow Blobs**: Dark circles under units for grounding
- **Hover/Select Effects**: Interactive feedback with glowing and rotation

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

## Controls

- **Left Click + Drag**: Rotate camera
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Click Unit**: Select unit (shows yellow indicator)
- **Hover Unit**: Highlights unit with glow

## Build

```bash
npm run build
```

## Technical Details

- **Renderer**: Three.js with ACES Filmic tone mapping
- **Tone Mapping Exposure**: 1.2 for brighter overall scene
- **Shadow Resolution**: 2048x2048 for crisp shadows
- **Anti-aliasing**: Enabled for smooth edges
- **Physics**: Units follow terrain height using procedural heightmap function

## Addressed Issues

1. ✅ Increased ambient light from dark to well-lit
2. ✅ Added environmental objects (trees, rocks, bushes, grass)
3. ✅ Improved unit colors with high contrast and emissive materials
4. ✅ Added ground texture variety with vertex colors
5. ✅ Fixed shadow rendering with proper shadow maps and lighting setup
