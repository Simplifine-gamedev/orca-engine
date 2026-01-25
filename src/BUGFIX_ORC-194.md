# Bug Fix: Units T-posing When Coming Out of Buildings (ORC-194)

## Issue Summary
Units were appearing in T-pose (default mesh pose) when spawning from buildings, before transitioning to their idle animation. This created an unprofessional appearance and broke immersion.

## Root Cause
The bug occurred due to three related issues:

1. **Missing Initial Animation State**: Units were created without an animation state set initially
2. **Animation Delay**: There was a gap between unit creation and animation initialization
3. **No Spawn Animation**: Units jumped directly from T-pose to idle animation

## Solution Implemented

### 1. Animation State Initialization (`src/store/gameStore.ts`)

**Before:**
```typescript
const newUnit: Unit = {
  id: unitId,
  position: { ...building.spawnPoint },
  type: unitType,
  health: 100,
  maxHealth: 100,
  // animationState was undefined or set after creation
};
```

**After:**
```typescript
const newUnit: Unit = {
  id: unitId,
  position: { ...building.spawnPoint },
  type: unitType,
  health: 100,
  maxHealth: 100,
  animationState: 'spawning', // ✓ Set immediately
  isSpawning: true,
};
```

### 2. Spawning Animation State (`src/types/unit.ts`)

Added 'spawning' as a distinct animation state:
```typescript
export type AnimationState = 'idle' | 'walking' | 'attacking' | 'dying' | 'spawning';
```

### 3. Immediate Animation Playback (`src/units/RTSUnit.tsx`)

The animation system now:
- Plays animation immediately on mount
- Uses fadeIn (200ms) for smooth transitions
- Has no gaps between animations
- Auto-transitions from 'spawning' to 'idle' after 500ms

```typescript
useEffect(() => {
  // Play animation immediately to prevent T-pose
  playAnimation(unit.animationState);
}, []);
```

## Testing Instructions

1. **Create a building** with a spawn point
2. **Spawn a unit** using `spawnUnit(buildingId, unitType)`
3. **Observe**: Unit should show spawning animation immediately
4. **After 500ms**: Unit should smoothly transition to idle animation
5. **Expected**: No T-pose should be visible at any point

## Visual Flow

```
Building spawns unit
         ↓
Unit created with animationState='spawning'
         ↓
RTSUnit component mounts
         ↓
Animation plays immediately (0ms delay)
         ↓
Spawning animation plays for 500ms
         ↓
Smooth transition to idle animation
```

## Benefits

1. **No T-pose visible**: Animation plays from frame 0
2. **Professional appearance**: Smooth spawn effect
3. **Better UX**: Clear visual feedback when units spawn
4. **Maintainable**: Clear animation state machine

## Files Modified

- `src/types/unit.ts` - Added spawning animation state
- `src/store/gameStore.ts` - Initialize units with spawning state
- `src/units/RTSUnit.tsx` - Immediate animation playback
- `src/README.md` - Documentation
- `src/example/GameExample.tsx` - Usage example

## Performance Impact

Minimal - the fix only adds:
- One additional animation state ('spawning')
- 500ms transition timer per spawned unit
- No impact on render performance

## Future Enhancements

Potential improvements:
- Customizable spawn animation duration per unit type
- Particle effects during spawning
- Sound effects synchronized with spawn animation
- Building-specific spawn animations
