# Orca RTS - Shadow Rendering Demo

This is a demonstration of proper shadow rendering in a 3D RTS game using React Three Fiber.

## Problem Solved

Previously, most models didn't show shadows consistently. This demo implements the correct shadow configuration.

## Shadow Configuration

### 1. Canvas Setup
- Shadows enabled on the Canvas component
- High-performance WebGL renderer

### 2. Lighting Setup (App.tsx)
- **DirectionalLight** with shadows enabled:
  - `castShadow={true}` - Light casts shadows
  - `shadow-mapSize={[2048, 2048]}` - High quality shadow map
  - Proper shadow camera bounds for coverage
  - Shadow bias to reduce artifacts

### 3. Model Configuration

#### RTSUnit Component (src/units/RTSUnit.tsx)
- All meshes have `castShadow={true}` - Units cast shadows
- All meshes have `receiveShadow={true}` - Units receive shadows from buildings and other units

#### Building Component (src/buildings/Building.tsx)
- All meshes have `castShadow={true}` - Buildings cast shadows
- All meshes have `receiveShadow={true}` - Buildings receive shadows

### 4. Ground Plane
- `receiveShadow={true}` - Ground receives all shadows

## Key Fixes Applied

1. ✅ All meshes have `castShadow={true}`
2. ✅ All meshes have `receiveShadow={true}`
3. ✅ Shadow map size increased to 2048x2048
4. ✅ DirectionalLight shadow settings properly configured
5. ✅ Shadow camera bounds set appropriately
6. ✅ Shadow bias configured to reduce artifacts

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the demo.

## Build

```bash
npm run build
```

## Technical Details

- **Framework**: React 18 + Vite
- **3D Library**: Three.js via React Three Fiber
- **Shadow Type**: PCF (Percentage Closer Filtering) soft shadows
- **Shadow Resolution**: 2048x2048 pixels

## Performance Notes

- Shadow map size of 2048x2048 provides good quality while maintaining performance
- For production, consider adjusting shadow map size based on target hardware
- DirectionalLight is used for consistent outdoor lighting and shadows
