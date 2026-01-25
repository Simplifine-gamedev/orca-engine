# Shadow Fix Guide - Linear Issue ORC-169

## Problem Description

Most models in the RTS game were not showing shadows consistently. Only some models would occasionally cast or receive shadows, leading to an inconsistent visual experience.

## Root Cause Analysis

The shadow inconsistency was caused by:

1. **Missing `castShadow` property** on mesh components
2. **Missing `receiveShadow` property** on mesh components
3. **Inadequate shadow map resolution**
4. **Improperly configured directional light shadow settings**

## Solution Implementation

### 1. Enable Shadows on Canvas

```tsx
<Canvas shadows>
  {/* ... */}
</Canvas>
```

The `shadows` prop enables shadow rendering in the Three.js WebGLRenderer.

### 2. Configure DirectionalLight Properly

```tsx
<directionalLight
  position={[10, 20, 10]}
  intensity={1.5}
  castShadow                          // Enable shadow casting
  shadow-mapSize={[2048, 2048]}       // High quality shadow map
  shadow-camera-far={50}              // How far shadows render
  shadow-camera-left={-20}            // Shadow camera bounds
  shadow-camera-right={20}
  shadow-camera-top={20}
  shadow-camera-bottom={-20}
  shadow-bias={-0.0001}              // Reduce shadow acne
/>
```

**Key parameters:**
- `castShadow`: Enables the light to cast shadows
- `shadow-mapSize`: Resolution of the shadow map (higher = better quality)
- `shadow-camera-*`: Defines the orthographic camera used for shadow rendering
- `shadow-bias`: Prevents shadow artifacts ("shadow acne")

### 3. Enable Shadow Properties on All Meshes

#### Units (RTSUnit.tsx)

Every mesh in the unit must have both properties:

```tsx
<mesh castShadow receiveShadow>
  <boxGeometry args={[0.8, 0.4, 1.0]} />
  <meshStandardMaterial color={color} />
</mesh>
```

- `castShadow={true}`: The mesh casts shadows onto other objects
- `receiveShadow={true}`: The mesh receives shadows from other objects

#### Buildings (Building.tsx)

Same configuration for all building meshes:

```tsx
<mesh castShadow receiveShadow>
  <boxGeometry args={[2, 2, 2]} />
  <meshStandardMaterial color={color} />
</mesh>
```

#### Ground Plane

The ground must receive shadows:

```tsx
<mesh receiveShadow>
  <planeGeometry args={[50, 50]} />
  <meshStandardMaterial color="#2d5016" />
</mesh>
```

Note: The ground typically doesn't need `castShadow` as there's nothing below it.

## Testing the Fix

1. **Install dependencies**:
   ```bash
   cd rts-game
   npm install
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Visual verification**:
   - All units should cast shadows on the ground
   - All buildings should cast shadows on the ground
   - Units near buildings should receive building shadows
   - Shadows should move smoothly with the light direction
   - No flickering or shadow artifacts

## Performance Considerations

### Shadow Map Resolution

| Resolution | Quality | Performance Impact |
|------------|---------|-------------------|
| 512x512    | Low     | Minimal          |
| 1024x1024  | Medium  | Low              |
| 2048x2048  | High    | Medium (Recommended) |
| 4096x4096  | Very High | High           |

**Recommendation**: Use 2048x2048 for most applications. Adjust based on target hardware.

### Shadow Camera Bounds

Set shadow camera bounds to match your scene:
- Too large: Shadows lose detail
- Too small: Objects outside bounds won't cast shadows

```tsx
shadow-camera-left={-20}
shadow-camera-right={20}
shadow-camera-top={20}
shadow-camera-bottom={-20}
```

### Multiple Lights

Only enable shadows on lights that need them:
- **DirectionalLight** (sun): Enable shadows
- **AmbientLight**: Cannot cast shadows (general illumination)
- **PointLight** (lamps): Enable shadows only if needed (expensive)

## Common Pitfalls

### ❌ Forgetting to enable shadows on Canvas
```tsx
<Canvas>  {/* Missing shadows prop */}
```

### ✅ Correct
```tsx
<Canvas shadows>
```

---

### ❌ Only setting castShadow without receiveShadow
```tsx
<mesh castShadow>  {/* Won't receive shadows from other objects */}
```

### ✅ Correct
```tsx
<mesh castShadow receiveShadow>
```

---

### ❌ Using MeshBasicMaterial
```tsx
<meshBasicMaterial color="#ff0000" />  {/* Doesn't support shadows */}
```

### ✅ Correct
```tsx
<meshStandardMaterial color="#ff0000" />  {/* Supports shadows */}
```

---

### ❌ Forgetting shadow-mapSize
```tsx
<directionalLight castShadow />  {/* Low default resolution */}
```

### ✅ Correct
```tsx
<directionalLight castShadow shadow-mapSize={[2048, 2048]} />
```

## Results

After implementing these fixes:

✅ All units consistently cast shadows  
✅ All buildings consistently cast shadows  
✅ All models receive shadows from other objects  
✅ Shadows are high quality (2048x2048 resolution)  
✅ No shadow artifacts or flickering  
✅ Consistent visual experience across all models  

## References

- [Three.js Shadow Documentation](https://threejs.org/docs/#api/en/lights/shadows/LightShadow)
- [React Three Fiber Shadows](https://docs.pmnd.rs/react-three-fiber/api/canvas#shadows)
- [Shadow Mapping Techniques](https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping)
