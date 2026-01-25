# Orca RTS - Population Counter Fix

## Bug Fix: ORC-138

### Problem
The population counter was showing the total world population instead of just the player's faction population.

### Solution
Fixed the `ResourceBar` component to use `getPlayerPopulation()` instead of `getWorldPopulation()`.

## File Structure

```
src/
├── store/
│   └── gameStore.ts      # Game state management with Zustand
├── ui/
│   ├── ResourceBar.tsx   # Population counter UI component (FIXED)
│   ├── ResourceBar.css   # Styles for ResourceBar
│   └── __tests__/
│       └── ResourceBar.test.tsx  # Unit tests
└── README.md
```

## Usage

### Setting up the game store

```typescript
import { useGameStore } from './store/gameStore';

// Initialize factions
const playerFaction = {
  id: 'player1',
  name: 'Blue Team',
  color: '#0066CC',
  isPlayer: true
};

// Set player faction
useGameStore.getState().setPlayerFaction(playerFaction.id);

// Add units
useGameStore.getState().addUnit({
  id: 'unit1',
  type: 'warrior',
  factionId: 'player1',
  health: 100,
  maxHealth: 100
});
```

### Using the ResourceBar component

```tsx
import { ResourceBar } from './ui/ResourceBar';
import './ui/ResourceBar.css';

function Game() {
  return (
    <div className="game-container">
      <ResourceBar />
      {/* Rest of your game UI */}
    </div>
  );
}
```

## Key Changes

### Before (Bug)
The component was incorrectly using world population:
```typescript
const population = useGameStore(state => state.getWorldPopulation());
```

### After (Fixed)
Now correctly uses player faction population:
```typescript
const playerPopulation = useGameStore(state => state.getPlayerPopulation());
const maxPopulation = useGameStore(state => state.getPlayerMaxPopulation());
```

## Population Calculation

The game store now provides these methods:

- `getPlayerPopulation()` - Returns count of units belonging to the player's faction
- `getPlayerMaxPopulation()` - Returns the maximum population limit for the player
- `getWorldPopulation()` - Returns total count of all units across all factions

## Display Format

The population counter now shows:
```
👥 [Your Units] / [Your Max Population]
```

For example: `👥 25 / 100`

This clearly shows only the player's faction population, not the total world population.

## Testing

Run the unit tests to verify the fix:

```bash
npm test -- ResourceBar.test.tsx
```

The tests verify that:
1. Player population is displayed (not world population)
2. Faction information is shown correctly
3. Zero population is handled properly
