# Orca RTS Game

RTS game built with Orca Engine featuring proper unit animation state management.

## T-Pose Bug Fix (ORC-194)

### Problem
Units were appearing in T-pose when spawning from buildings before transitioning to idle animation.

### Solution
The fix involves three key changes:

1. **Immediate Animation State Initialization** (`gameStore.ts`)
   - Units now spawn with `animationState: 'spawning'` instead of undefined
   - This ensures an animation is set from the moment the unit is created

2. **Spawning Animation State** (`RTSUnit.tsx`)
   - Added 'spawning' to the animation state machine
   - Animation plays immediately on component mount
   - Smooth transition from 'spawning' to 'idle' after 500ms

3. **Animation State Machine** (`RTSUnit.tsx`)
   - Proper fadeIn/fadeOut transitions between animations
   - No gaps where T-pose could appear
   - Immediate playback with 200ms fade-in

### Architecture

```
src/
├── types/
│   └── unit.ts          # Type definitions for units and buildings
├── store/
│   └── gameStore.ts     # Zustand store with unit spawning logic
└── units/
    └── RTSUnit.tsx      # Unit component with animation state machine
```

### Usage Example

```typescript
import { useGameStore } from './store/gameStore';
import { RTSUnit } from './units/RTSUnit';

function Game() {
  const { units, spawnUnit } = useGameStore();

  const handleSpawnUnit = () => {
    spawnUnit('barracks-1', 'soldier');
  };

  return (
    <div className="game-container">
      {Array.from(units.values()).map(unit => (
        <RTSUnit key={unit.id} unit={unit} />
      ))}
      <button onClick={handleSpawnUnit}>Spawn Unit</button>
    </div>
  );
}
```

### Animation States

- `spawning`: Brief animation when unit first appears (500ms)
- `idle`: Default looping animation when unit is not doing anything
- `walking`: Looping animation while unit moves
- `attacking`: One-shot animation that returns to idle
- `dying`: One-shot animation before unit removal

### Testing the Fix

1. Spawn a unit from a building
2. Verify unit shows 'spawning' animation immediately
3. After 500ms, verify smooth transition to 'idle'
4. No T-pose should be visible at any point

### Dependencies

```json
{
  "zustand": "^4.0.0",
  "react": "^18.0.0",
  "three": "^0.150.0" // For 3D model rendering
}
```
