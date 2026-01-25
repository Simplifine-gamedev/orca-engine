# Quick Integration Guide - Wall System

## 🚀 Quick Start (3 Steps)

### Step 1: Initialize on Game Start

```typescript
import { initializeWallSystem } from './buildings';

// In your main game initialization
async function startGame() {
  await initializeWallSystem();
  console.log('Wall system ready!');
}
```

### Step 2: Add to Your UI

```typescript
import { useWallSystem, WallLoadingIndicator } from './buildings';

function GameUI() {
  const { 
    isBuildMode, 
    isPreloading, 
    enterBuildMode, 
    exitBuildMode 
  } = useWallSystem();

  return (
    <>
      <WallLoadingIndicator isLoading={isPreloading} />
      
      <button onClick={enterBuildMode} disabled={isPreloading}>
        Build Walls
      </button>
      
      {isBuildMode && (
        <button onClick={exitBuildMode}>Exit</button>
      )}
    </>
  );
}
```

### Step 3: Render Previews

```typescript
import { WallPreview } from './buildings';

function GameScene() {
  return (
    <Canvas>
      <WallPreview 
        position={[0, 0, 0]} 
        type="wall_segment"
        isValid={true}
      />
    </Canvas>
  );
}
```

## ✅ That's It!

Your wall system is now optimized with:
- ⚡ Instant build mode activation
- 💾 Cached geometries and materials
- 🎨 Loading indicators
- 🚀 97-99% performance improvement

## 📚 More Information

- See [README.md](./buildings/README.md) for detailed API reference
- See [PERFORMANCE.md](./buildings/PERFORMANCE.md) for benchmarks
- See [example.tsx](./buildings/example.tsx) for complete example

## 🐛 Troubleshooting

**Build mode is slow?**
- Make sure you called `initializeWallSystem()` on game start

**No loading indicator?**
- Check that `<WallLoadingIndicator>` is rendered
- Verify `isPreloading` state is passed correctly

**Missing previews?**
- Ensure Three.js and React Three Fiber are installed
- Check browser console for errors

## 📦 Dependencies

Required packages:
- `three` (^0.160.0)
- `react` (^18.2.0)
- `@react-three/fiber` (^8.15.0)

Install with:
```bash
npm install three react @react-three/fiber
```

---

For questions or issues, please refer to the full documentation in the `buildings/` directory.
