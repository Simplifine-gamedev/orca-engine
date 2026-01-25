# Testing Guide - Shadow Fix for ORC-169

## Quick Start

### 1. Install Dependencies

```bash
cd rts-game
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

The application will start at `http://localhost:3000`

## What to Test

### Visual Verification Checklist

✅ **Units Cast Shadows**
- Rotate the camera around the scene
- Verify each of the 3 colored units (red, teal, blue) casts a visible shadow on the ground
- Shadows should be crisp and consistent

✅ **Buildings Cast Shadows**
- All 3 buildings (gray structures with red roofs) should cast shadows
- Building shadows should be larger and more prominent than unit shadows

✅ **Objects Receive Shadows**
- Move the camera to see units positioned near buildings
- Units should show building shadows on their surfaces
- Buildings can receive shadows from other nearby buildings

✅ **Ground Receives All Shadows**
- The green ground plane should show shadows from all units and buildings
- Shadows should be properly positioned relative to the light source

✅ **Shadow Quality**
- Shadows should have soft edges (PCF filtering)
- No flickering or "shadow acne" artifacts
- No gaps or missing shadows

### Interactive Testing

1. **Orbit Controls**: Click and drag to rotate the camera
2. **Zoom**: Scroll to zoom in/out
3. **Pan**: Right-click and drag to pan

### Browser Console

Open browser DevTools (F12) and check:
- No WebGL errors
- Stats panel shows FPS (should be 60fps on modern hardware)

## Expected Results

### Before Fix (Issue ORC-169)
❌ Most models didn't show shadows  
❌ Inconsistent shadow rendering  
❌ Only some objects would occasionally cast shadows  

### After Fix (Current State)
✅ All 3 units consistently cast shadows  
✅ All 3 buildings consistently cast shadows  
✅ All models receive shadows from other objects  
✅ High-quality shadows (2048x2048 resolution)  
✅ No visual artifacts  

## Performance Metrics

On modern hardware, expect:
- **FPS**: 60fps (constant)
- **Memory**: ~150-200MB
- **GPU Usage**: Low to Medium

If performance is poor:
1. Reduce shadow map size in `src/App.tsx`:
   ```tsx
   shadow-mapSize={[1024, 1024]}  // Instead of [2048, 2048]
   ```

2. Reduce number of objects casting shadows

## Troubleshooting

### Problem: No shadows visible

**Solution**: Ensure browser supports WebGL 2
- Check: https://get.webgl.org/webgl2/

### Problem: Shadows are blocky/pixelated

**Solution**: Increase shadow map size in `src/App.tsx`:
```tsx
shadow-mapSize={[4096, 4096]}
```

### Problem: Performance issues

**Solution**: Reduce shadow map size:
```tsx
shadow-mapSize={[1024, 1024]}
```

### Problem: Shadow artifacts (shadow acne)

**Solution**: Adjust shadow bias in `src/App.tsx`:
```tsx
shadow-bias={-0.0005}  // Increase absolute value
```

## Build for Production

```bash
npm run build
```

Optimized build will be in `dist/` directory.

## Technical Validation

### Code Review Checklist

- [ ] All mesh components have `castShadow={true}`
- [ ] All mesh components have `receiveShadow={true}`
- [ ] Canvas has `shadows` prop enabled
- [ ] DirectionalLight has `castShadow={true}`
- [ ] Shadow map size is set to at least 1024x1024
- [ ] Shadow camera bounds cover all objects
- [ ] Using `meshStandardMaterial` (not `meshBasicMaterial`)

## References

- Main implementation: `src/App.tsx`
- Unit component: `src/units/RTSUnit.tsx`
- Building component: `src/buildings/Building.tsx`
- Detailed guide: `SHADOW_FIX_GUIDE.md`
