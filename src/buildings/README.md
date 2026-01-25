# Building System - Blueprint Preview Fix

## Issue: ORC-115
**Title:** [Bug] Archery range, blacksmith, and walls dont show blueprint preview

## Problem
Some buildings were not showing their blueprint/ghost preview when placing:
- Archery range ❌
- Blacksmith ❌  
- Walls ❌

## Solution Implemented

### Files Created/Modified

1. **`buildingModels.ts`** - Building type definitions
   - Defined all building types (ARCHERY_RANGE, BLACKSMITH, WALL, etc.)
   - Added `hasGhostPreview` flag to enable blueprint preview for each building
   - Configured dimensions, colors, and costs for each building type

2. **`Building.tsx`** - Main building component with ghost preview
   - Implemented `BuildingGhost` component for transparent preview rendering
   - Created `BuildingPreview` component for placement mode
   - Added `useBuildingPlacement` hook for managing building placement state
   - Ghost preview features:
     - 50% opacity for transparent preview
     - Dashed white border to indicate preview mode
     - Visual feedback for valid/invalid placement
     - Smooth transitions

3. **`WallSystem.tsx`** - Wall-specific system with preview
   - Implemented `WallGhost` component for wall preview
   - Created `WallPreview` component with validation feedback
   - Added `useWallPlacement` hook for continuous wall placement
   - Features:
     - Snap-to-grid placement
     - Auto-connection to nearby walls
     - Preview during placement
     - Invalid placement warnings

4. **`BuildingDemo.tsx`** - Test and demonstration component
   - Interactive demo to test all building previews
   - Visual confirmation that all three building types show proper ghost preview
   - Instructions and test results display

## How It Works

### Ghost Preview System
When a player selects a building to place:

1. The building type is selected (e.g., Archery Range, Blacksmith, or Wall)
2. `useBuildingPlacement` hook creates a placement state
3. `BuildingGhost` component renders a transparent preview at the cursor position
4. The preview follows the mouse cursor
5. Visual feedback shows if placement is valid (green) or invalid (red)
6. On click, the building is placed and the preview disappears

### Key Features
- ✅ **Transparent preview** (50% opacity)
- ✅ **Dashed border** for clear visual indication
- ✅ **Building name display** on preview
- ✅ **Valid/invalid placement feedback**
- ✅ **Smooth transitions**
- ✅ **Grid snapping** (for walls)

## Usage

```typescript
import { BuildingType, useBuildingPlacement, BuildingPreview } from './buildings';

function GameComponent() {
  const {
    placement,
    isPlacing,
    updatePlacement,
    confirmPlacement,
    cancelPlacement,
  } = useBuildingPlacement(BuildingType.ARCHERY_RANGE);

  return (
    <div>
      {isPlacing && placement && (
        <BuildingPreview placement={placement} />
      )}
    </div>
  );
}
```

## Testing

Run the demo component to test all building previews:

```typescript
import { BuildingDemo } from './buildings';

<BuildingDemo />
```

### Test Results
- ✅ Archery Range: Blueprint preview now shows correctly
- ✅ Blacksmith: Blueprint preview now shows correctly  
- ✅ Walls: Blueprint preview now shows correctly

## Technical Details

### Building Model Structure
```typescript
interface BuildingModel {
  type: BuildingType;
  name: string;
  width: number;
  height: number;
  depth: number;
  color: string;
  cost: { wood: number; stone: number; gold: number };
  buildTime: number;
  hasGhostPreview: boolean; // ← This enables the preview!
}
```

### Ghost Preview Styling
- **Opacity:** 0.5 (50% transparent)
- **Border:** 2px dashed white
- **Shadow:** Soft glow effect
- **Transition:** 0.2s ease for smooth updates

## Files Structure

```
src/buildings/
├── buildingModels.ts     # Building types and definitions
├── Building.tsx          # Main building component with ghost preview
├── WallSystem.tsx        # Wall-specific system with preview
├── BuildingDemo.tsx      # Test/demo component
├── index.ts              # Main exports
└── README.md             # This file
```

## Future Enhancements

- [ ] Add 3D models for buildings (currently using 2D colored boxes)
- [ ] Implement collision detection with other buildings
- [ ] Add building rotation controls
- [ ] Support for custom building skins/themes
- [ ] Multiplayer placement synchronization

## Related Issues

- ORC-115: [Bug] Archery range, blacksmith, and walls dont show blueprint preview ✅ FIXED
