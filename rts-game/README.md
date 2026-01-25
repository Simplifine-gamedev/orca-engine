# Orca RTS - Building Preview Bug Fix

This project demonstrates and fixes the bug reported in Linear issue ORC-102.

## Bug Description

**Problem:** When a worker is actively mining a gold mine and the player tries to place a building (like barracks), the building preview/model does not show up.

**Root Cause:** The `BuildingGhost` component was incorrectly checking the worker state before rendering the building preview. This caused the ghost to not appear when any worker was in a mining state.

## The Fix

### Before (Buggy Code)

```typescript
export function BuildingGhost() {
  const buildingPlacement = useGameStore((state) => state.buildingPlacement);
  const workers = useGameStore((state) => state.workers);
  
  // BUG: Checking worker state when we shouldn't
  const hasWorkerMining = workers.some((w) => w.state === 'mining');
  
  // This prevents the ghost from showing if any worker is mining
  if (!buildingPlacement.isActive || !buildingPlacement.ghostPosition || hasWorkerMining) {
    return null;
  }
  
  // ... rest of component
}
```

### After (Fixed Code)

```typescript
export function BuildingGhost() {
  const buildingPlacement = useGameStore((state) => state.buildingPlacement);
  
  // FIX: Only check if building placement is active and has a ghost position
  // The building ghost should show regardless of worker state
  if (!buildingPlacement.isActive || !buildingPlacement.ghostPosition || !buildingPlacement.type) {
    return null;
  }
  
  // ... rest of component
}
```

## Key Changes

1. **Removed worker state dependency**: The building ghost no longer checks if any worker is mining
2. **Decoupled systems**: Building placement and worker actions are now properly independent
3. **Fixed visibility logic**: Ghost only depends on placement state, not worker state

## Project Structure

```
rts-game/
├── src/
│   ├── buildings/
│   │   └── Building.tsx       # Building and BuildingGhost components (FIX HERE)
│   ├── components/
│   │   ├── Worker.tsx         # Worker component with mining animation
│   │   └── Resource.tsx       # Gold mine and resource components
│   ├── store/
│   │   └── gameStore.ts       # Zustand state management
│   ├── App.tsx                # Main app with interaction handlers
│   ├── main.tsx               # Entry point
│   └── index.css              # Styles
├── package.json
├── tsconfig.json
└── README.md
```

## How to Test the Fix

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Run the development server:**
   ```bash
   npm run dev
   ```

3. **Reproduce the original bug scenario:**
   - Select a worker (click on the blue capsule)
   - Click "Start Mining" to set the worker to mining state
   - Click "Build Barracks" or any building button
   - **Result:** Building preview now appears correctly! ✓

4. **Verify the fix:**
   - The building ghost (green wireframe) should appear and follow your cursor
   - This works regardless of whether workers are mining, idle, or in any other state
   - Click to place the building, right-click to cancel

## Technologies Used

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Three.js** - 3D rendering
- **@react-three/fiber** - React renderer for Three.js
- **@react-three/drei** - Useful helpers for R3F
- **Zustand** - State management
- **Vite** - Build tool

## Implementation Details

### State Management (gameStore.ts)

The game state is managed using Zustand with the following key features:
- Worker state tracking (idle, mining, building, moving)
- Building placement mode with ghost position
- Resource management (gold mines)
- Building registry

### Building Placement Flow

1. User clicks a build button → `startBuildingPlacement()` is called
2. Mouse moves over ground → `updateBuildingGhostPosition()` updates ghost position
3. BuildingGhost component renders preview at ghost position
4. User clicks to place → `confirmBuildingPlacement()` creates the building
5. User right-clicks to cancel → `cancelBuildingPlacement()` clears placement mode

### Worker System

Workers have different states that affect their appearance:
- **Idle** - Blue color, standing still
- **Mining** - Gold color, bobbing animation
- **Building** - Brown color
- **Moving** - Light blue color

## Files Changed for Bug Fix

**Primary Fix:**
- `src/buildings/Building.tsx` - Removed incorrect worker state check in `BuildingGhost` component

**Supporting Implementation:**
- `src/store/gameStore.ts` - Building placement state management
- `src/App.tsx` - Ground interaction handlers for building placement
- `src/components/Worker.tsx` - Worker rendering and state visualization
- `src/components/Resource.tsx` - Resource (gold mine) rendering

## Testing Checklist

- [x] Building ghost appears when placement mode is active
- [x] Ghost follows cursor correctly
- [x] Ghost shows while worker is idle
- [x] Ghost shows while worker is mining (BUG FIX)
- [x] Ghost shows while multiple workers are in different states
- [x] Building can be placed successfully
- [x] Building placement can be cancelled
- [x] Worker state changes don't affect building placement

## Deployment

To build for production:

```bash
npm run build
```

The production build will be in the `dist/` directory.

## License

MIT
