# Bug Fix: Human Barracks Shows Dwarf Unit Previews

## Issue: ORC-114

### Problem Description
When playing as humans, the barracks was showing dwarf unit previews/thumbnails, but actually spawned human units correctly.

### Root Cause
The `Building.tsx` component was using a hardcoded faction ID ('dwarf') when fetching unit previews, instead of using the player's actual faction ID.

**Buggy Code (BEFORE):**
```typescript
// In Building.tsx
const faction = getFaction('dwarf'); // WRONG: Hardcoded to dwarf faction
```

This meant that regardless of which faction the player was playing, the building would always show dwarf unit previews.

### Solution
Pass the `playerFactionId` prop to the Building component and use it consistently when fetching faction data and unit previews.

**Fixed Code (AFTER):**
```typescript
// In Building.tsx
const faction = getFaction(playerFactionId); // CORRECT: Uses player's actual faction
```

### Changes Made

#### 1. `src/config/factions.ts`
- Created comprehensive faction configuration system
- Added helper functions: `getFaction()`, `getUnitPreview()`, `getBuildingUnits()`
- Defined both human and dwarf factions with their respective units and preview images

#### 2. `src/buildings/Building.tsx`
- **KEY FIX**: Changed from hardcoded `getFaction('dwarf')` to `getFaction(playerFactionId)`
- Added `playerFactionId` prop to BuildingProps interface
- Used `getUnitPreview(playerFactionId, unitId)` to fetch correct preview images
- Added clear comments explaining the bug and fix

#### 3. `src/ui/SelectionPanel.tsx`
- Properly passes `playerFactionId` to Building component
- Validates that selected entity belongs to player
- Uses correct faction data for all UI elements

### Testing the Fix

**Before Fix:**
1. Start game as Human player
2. Select barracks
3. ❌ **Bug**: Shows dwarf unit previews (Warrior, Rifleman, Hammerer)
4. ✅ Spawns correct human units (Footman, Archer, Knight)

**After Fix:**
1. Start game as Human player
2. Select barracks
3. ✅ Shows correct human unit previews (Footman, Archer, Knight)
4. ✅ Spawns correct human units (Footman, Archer, Knight)

### Key Takeaways

**Anti-Pattern (Don't Do This):**
```typescript
// WRONG: Hardcoding faction IDs
const faction = getFaction('dwarf');
const preview = '/assets/units/dwarf/warrior_preview.png';
```

**Best Practice (Do This):**
```typescript
// CORRECT: Use dynamic faction IDs from props/state
const faction = getFaction(playerFactionId);
const preview = getUnitPreview(playerFactionId, unitId);
```

### Prevention

To prevent similar bugs in the future:

1. **Never hardcode faction IDs** - always pass them as props or get from game state
2. **Type safety** - Use TypeScript interfaces to enforce faction ID parameters
3. **Code review** - Watch for hardcoded strings like 'dwarf', 'human', etc.
4. **Testing** - Test all features with each playable faction
5. **Centralized config** - Keep all faction data in `factions.ts` config file

### User Feedback

> "human barracks is showing the previews of drawven units…. Weird bug. But it spawns the human units"
> - Haridzieko

This bug has been resolved. Human barracks now correctly shows human unit previews.
