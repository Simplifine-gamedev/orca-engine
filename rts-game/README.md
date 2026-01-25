# Orca RTS - Asset Preloading Solution

## Problem (ORC-100)

Models (especially workers) were taking too long to load when the game starts, making the game feel buggy and laggy.

**User Feedback:**
- Gaudio: "took quite a bit of time for models (especially the workers) to load"
- "When game starts, some of the models aren't already loaded... It takes a while and feels buggy/laggy"

## Solution

This implementation provides a complete asset preloading system that:

1. **Preloads all critical assets during the loading screen**
2. **Shows loading progress for each asset type** (models, textures, audio)
3. **Only starts the game when critical assets are loaded**
4. **Uses GLTF caching to prevent re-downloading** same models

## Key Features

### 1. AssetPreloader System (`src/systems/AssetPreloader.ts`)

A robust asset preloading system that:
- Loads models, textures, and audio files
- Provides progress callbacks for UI updates
- Caches GLTF models to avoid re-downloading
- Marks assets as critical or optional
- Handles errors gracefully

```typescript
// Example usage
assetPreloader.registerAssets([
  {
    name: 'worker',
    url: '/models/worker.glb',
    type: 'model',
    critical: true, // Game won't start without this
  },
]);

await assetPreloader.preloadAll();
```

### 2. LoadingOverlay Component (`src/ui/LoadingOverlay.tsx`)

A visual loading screen that:
- Shows overall loading percentage
- Displays current asset being loaded
- Color-codes asset types (models, textures, audio)
- Shows error messages if loading fails
- Prevents interaction until loading is complete

### 3. Unit Components with Preloaded Assets

#### NeutralMob (`src/units/NeutralMob.tsx`)
- Uses preloaded models from the asset preloader
- Falls back to placeholder if model isn't loaded
- Implements simple idle animations

#### RTSUnit (`src/units/RTSUnit.tsx`)
- Player-controlled units with preloaded models
- Team-colored materials
- Different behaviors for worker/soldier/builder types
- Click-to-select functionality

### 4. GLTF Cache

```typescript
export const gltfCache = new Map<string, GLTF>();
```

The cache prevents re-downloading the same model multiple times. When a model is loaded:
1. Check if it's in the cache
2. If yes, use the cached version (instant)
3. If no, load it and add to cache

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  App.tsx                        │
│  - Initializes game                             │
│  - Registers all assets                         │
│  - Handles preloading lifecycle                 │
└───────────────┬─────────────────────────────────┘
                │
                ├──────────────────┐
                │                  │
┌───────────────▼───────┐  ┌───────▼───────────────┐
│   AssetPreloader      │  │   LoadingOverlay      │
│  - Loads models       │  │  - Shows progress     │
│  - Loads textures     │  │  - Updates UI         │
│  - Loads audio        │  │  - Displays errors    │
│  - Tracks progress    │  │                       │
│  - Caches assets      │  │                       │
└───────────┬───────────┘  └───────────────────────┘
            │
            │ Provides preloaded assets
            │
┌───────────▼───────────────────────────────────────┐
│              Game Scene                           │
│  ┌─────────────┐  ┌─────────────┐                │
│  │  RTSUnit    │  │ NeutralMob  │                │
│  │  - Worker   │  │  - Creature │                │
│  │  - Soldier  │  │  - Resource │                │
│  │  - Builder  │  │             │                │
│  └─────────────┘  └─────────────┘                │
│  All models load INSTANTLY (already in memory!)  │
└───────────────────────────────────────────────────┘
```

## How It Fixes the Issues

### Before (Buggy/Laggy):
```
Game Start → User sees empty scene → Models load one by one →
User waits and sees models pop in → Feels buggy
```

### After (Smooth):
```
Game Start → Loading screen appears → All models preload →
Progress bar shows status → 100% complete → Game starts →
All models already in memory → Instant rendering
```

## Installation & Usage

### 1. Install Dependencies

```bash
cd rts-game
npm install
```

### 2. Add Your Models

Place your GLTF/GLB model files in `/public/models/`:
```
/public
  /models
    - worker.glb
    - soldier.glb
    - builder.glb
    - neutral_creature.glb
    - neutral_resource.glb
  /textures
    - terrain.png
    - ui.png
  /audio
    - background.mp3
    - select.mp3
```

### 3. Register Assets

In `src/App.tsx`, update the `registerGameAssets()` function with your actual asset URLs:

```typescript
const registerGameAssets = () => {
  registerRTSUnitAssets([
    {
      name: 'worker',
      url: '/models/worker.glb',
      critical: true,
    },
    // ... more assets
  ]);
};
```

### 4. Run the Game

```bash
npm run dev
```

Open http://localhost:3000 to see the game with asset preloading.

## Technical Details

### Asset Types

- **Models (GLTF/GLB)**: 3D models for units and objects
- **Textures (PNG/JPG)**: Images for materials and UI
- **Audio (MP3/OGG)**: Sound effects and music

### Critical vs Optional Assets

- **Critical**: Game won't start without these (e.g., player units)
- **Optional**: Game starts even if these fail (e.g., decorative elements)

### Progress Tracking

The `LoadingProgress` interface provides:
```typescript
{
  loaded: number;      // Number of assets loaded
  total: number;       // Total assets to load
  currentAsset: string; // Name of current asset
  percentage: number;   // 0-100 progress
  assetType: AssetType; // 'model' | 'texture' | 'audio'
}
```

### Error Handling

- Non-critical assets: Game continues, error logged
- Critical assets: Game stops, error shown to user

## Performance Benefits

1. **No Runtime Loading**: All models loaded before gameplay
2. **Caching**: Models loaded once, reused for all instances
3. **Instant Rendering**: Units appear immediately when spawned
4. **Better UX**: Users see progress, not broken-looking game

## Customization

### Adjust Loading Screen

Edit `src/ui/LoadingOverlay.tsx` to:
- Change colors and styling
- Add logo or branding
- Modify animation timing
- Add loading tips or hints

### Add More Asset Types

Extend `AssetPreloader.ts` to support:
- Fonts
- Shaders
- Particle systems
- Custom file formats

### Progress Callbacks

```typescript
assetPreloader.setProgressCallback((progress) => {
  // Custom progress handling
  console.log(`${progress.percentage}% - ${progress.currentAsset}`);
  
  // Update analytics
  analytics.track('asset_loading', { asset: progress.currentAsset });
});
```

## Testing

### Simulate Slow Loading

To test the loading screen with slow connections:

1. Open Chrome DevTools
2. Go to Network tab
3. Select "Slow 3G" from throttling dropdown
4. Refresh the page

You'll see the loading screen in action!

### Test Error Handling

To test error handling:

1. Change an asset URL to an invalid path
2. Mark it as critical
3. See how the error is displayed

## Migration Guide

If you have existing code without preloading:

### Old Code (No Preloading)
```typescript
function NeutralMob({ modelUrl }) {
  const gltf = useLoader(GLTFLoader, modelUrl); // Loads during render!
  return <primitive object={gltf.scene} />;
}
```

### New Code (With Preloading)
```typescript
function NeutralMob({ modelName }) {
  const gltf = assetPreloader.getModel(modelName); // Already loaded!
  return <primitive object={gltf.scene} />;
}

// In initialization:
registerNeutralMobAssets([
  { name: 'mob1', url: '/models/mob1.glb' }
]);
await assetPreloader.preloadAll();
```

## Future Enhancements

Potential improvements:
- [ ] Add streaming for large assets
- [ ] Implement progressive loading (critical first)
- [ ] Add asset compression
- [ ] Cache assets in IndexedDB for repeat visits
- [ ] Add memory management for mobile devices
- [ ] Implement LOD (Level of Detail) system

## References

- [Three.js GLTFLoader](https://threejs.org/docs/#examples/en/loaders/GLTFLoader)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)
- [Asset Loading Best Practices](https://web.dev/preload-critical-assets/)

## License

MIT

## Contributing

Issues mentioned this was needed for the Orca RTS game. Feel free to adapt this implementation for your specific needs.
