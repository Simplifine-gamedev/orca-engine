# Bug Fix: Release All Button Not Working

## Issue Description

The "Release all" button in buildings was not working correctly. When clicked, it would clear the building's garrison list but wouldn't update the garrisoned units themselves, leaving the game in an inconsistent state.

## Root Cause

The `releaseAllUnits` function in `src/store/gameStore.ts` had an incomplete implementation:

```typescript
// BEFORE (buggy code)
releaseAllUnits: (buildingId) =>
  set((state) => {
    const building = state.buildings[buildingId];
    
    if (!building) return state;

    return {
      buildings: {
        ...state.buildings,
        [buildingId]: {
          ...building,
          garrisonedUnits: [], // Only cleared the building's list
        },
      },
      // Missing: Update to units themselves!
    };
  }),
```

The function only updated the building's `garrisonedUnits` array but failed to:
1. Clear each unit's `garrisonedIn` property
2. Set new positions for the released units

This created a state inconsistency where:
- The building showed no garrisoned units
- But the units still had their `garrisonedIn` property set
- The units would not appear on the map

## The Fix

The corrected implementation now properly updates both the building AND all units:

```typescript
// AFTER (fixed code)
releaseAllUnits: (buildingId) =>
  set((state) => {
    const building = state.buildings[buildingId];
    
    if (!building || building.garrisonedUnits.length === 0) return state;

    // Create updated units with cleared garrison status and exit positions
    const updatedUnits = { ...state.units };
    
    building.garrisonedUnits.forEach((unitId, index) => {
      const unit = state.units[unitId];
      if (unit) {
        // Calculate exit position (spread units around the building)
        const angle = (index / building.garrisonedUnits.length) * Math.PI * 2;
        const radius = 60;
        const exitPosition = {
          x: building.position.x + Math.cos(angle) * radius,
          y: building.position.y + Math.sin(angle) * radius,
        };

        updatedUnits[unitId] = {
          ...unit,
          garrisonedIn: undefined,
          position: exitPosition,
        };
      }
    });

    return {
      units: updatedUnits,
      buildings: {
        ...state.buildings,
        [buildingId]: {
          ...building,
          garrisonedUnits: [],
        },
      },
    };
  }),
```

## Changes Made

1. **Proper State Update**: Now updates both `units` and `buildings` in the store
2. **Clear Garrison Status**: Removes `garrisonedIn` property from each unit
3. **Position Calculation**: Places released units in a circle around the building
4. **Edge Case Handling**: Checks for empty garrison before processing

## Testing

To test the fix:

1. Run the demo: `cd rts-game && npm install && npm run dev`
2. Click on the Castle (left building with 3 garrisoned units)
3. Click "Release All Units" button
4. Verify that:
   - The garrison panel closes
   - All 3 units appear on the map around the building
   - Units are positioned in a circle around the castle
   - Clicking the castle again shows 0 garrisoned units

## Files Modified

- `src/store/gameStore.ts` - Fixed the `releaseAllUnits` function

## Related Issue

Fixes Linear issue ORC-140: [Bug] Release all button not working in buildings
