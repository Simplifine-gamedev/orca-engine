# Orca RTS Demo

A simple Real-Time Strategy game demo showcasing buildings and garrisoned units.

## Bug Description

The "Release all" button in buildings is not working correctly. When clicked, it clears the building's garrison list but doesn't update the units themselves, leaving them in an inconsistent state.

## Files

- `src/buildings/Building.tsx` - Building component with garrison UI
- `src/store/gameStore.ts` - Game state management with the buggy `releaseAllUnits` function

## Running the Demo

```bash
cd rts-game
npm install
npm run dev
```

## Testing the Bug

1. Click on the Castle (brown building on the left)
2. You'll see 3 garrisoned units (Knight, Archer, Pikeman)
3. Click "Release All Units" button
4. Notice that the garrison panel disappears but the units don't appear on the map
5. The units are stuck in an inconsistent state

## The Fix

The `releaseAllUnits` function in `gameStore.ts` needs to:
1. Clear the building's `garrisonedUnits` array
2. Update each unit to remove their `garrisonedIn` property
3. Set exit positions for each unit
