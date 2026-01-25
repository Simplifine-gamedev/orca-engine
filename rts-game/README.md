# Orca RTS - Shadow Configuration Fixed

This project demonstrates the solution to **ORC-144**: Inconsistent shadows across RTS models.

## Problem Solved

Previously, most models didn't show shadows. This has been fixed by ensuring:

1. ✅ All meshes have `castShadow={true}`
2. ✅ All meshes have `receiveShadow={true}`
3. ✅ Shadow map settings are properly configured
4. ✅ Light shadow settings are optimized

## Shadow Implementation

### 1. Canvas Configuration (`src/App.tsx`)

```tsx
<Canvas
  shadows={{
    enabled: true,
    type: 'PCFSoftShadowMap', // Soft shadows for quality
  }}
>
```

### 2. Light Configuration (`src/App.tsx`)

The main directional light is configured with optimal shadow settings:

```tsx
<directionalLight
  position={[10, 15, 5]}
  intensity={1.2}
  castShadow
  shadow-mapSize-width={2048}
  shadow-mapSize-height={2048}
  shadow-camera-far={50}
  shadow-camera-left={-15}
  shadow-camera-right={15}
  shadow-camera-top={15}
  shadow-camera-bottom={-15}
  shadow-bias={-0.0001}
/>
```

### 3. Mesh Configuration

**Every mesh** in both `RTSUnit.tsx` and `Building.tsx` now has:

```tsx
<mesh castShadow receiveShadow>
  {/* geometry and material */}
</mesh>
```

This applies to:
- Unit bodies, heads, weapons, and bases
- Building structures, roofs, doors, windows, chimneys, and foundations
- Ground plane (receives shadows)

## Key Changes Made

### `src/App.tsx`
- Enabled shadows on Canvas with PCFSoftShadowMap
- Configured directional light with 2048x2048 shadow map
- Set proper shadow camera bounds (-15 to 15 in all directions)
- Added shadow-bias to prevent shadow acne
- Ground plane receives shadows

### `src/units/RTSUnit.tsx`
- **All 4 mesh components** now have `castShadow` and `receiveShadow`
- Main body (capsule)
- Head (sphere)
- Weapon/antenna (cylinder)
- Base platform (cylinder)

### `src/buildings/Building.tsx`
- **All 7 mesh components** now have `castShadow` and `receiveShadow`
- Main structure (box)
- Roof (cone)
- Door (box)
- Windows (2 boxes)
- Chimney (cylinder)
- Foundation (box)

## Performance Considerations

- Shadow map size: 2048x2048 (balance between quality and performance)
- Only the main light casts shadows (secondary fill light doesn't)
- PCFSoftShadowMap provides smooth shadows without excessive cost
- Shadow camera bounds are optimized for the scene size

## Running the Project

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Verification

When you run the project, you should now see:

1. All units casting shadows on the ground
2. Buildings casting shadows on the ground
3. Units and buildings receiving shadows from each other
4. Smooth shadow edges (PCF filtering)
5. No shadow artifacts (proper bias setting)

## Technical Details

- **Framework**: React + React Three Fiber
- **3D Library**: Three.js
- **Shadow Type**: PCFSoftShadowMap
- **Shadow Resolution**: 2048x2048
- **Light Type**: Directional (simulates sun)

## Issue Resolution

This implementation fully resolves **ORC-144** by ensuring consistent shadow behavior across all models in the RTS game. Every visible object now properly casts and receives shadows as expected.

---

**Status**: ✅ Fixed
**Issue**: ORC-144
**Date**: 2026-01-25
