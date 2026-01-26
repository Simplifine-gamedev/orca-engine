# Orca RTS - Bug Fix Implementation

This directory contains the corrected implementation for the **human barracks dwarf preview bug** (ORC-114).

## Bug Summary

**Issue**: Human barracks showed dwarf unit previews but spawned human units correctly.

**Fix**: Changed Building component to use `playerFactionId` instead of hardcoded 'dwarf' faction.

## File Structure

```
rts-game/
├── src/
│   ├── buildings/
│   │   └── Building.tsx          # Building UI with unit training
│   ├── ui/
│   │   └── SelectionPanel.tsx    # Main selection UI panel
│   ├── config/
│   │   └── factions.ts           # Faction configuration (humans, dwarves)
│   └── assets/                   # Asset directory (placeholders)
├── BUG_FIX_DOCUMENTATION.md      # Detailed bug analysis and fix
└── README.md                     # This file
```

## Key Changes

### Before (Buggy)
```typescript
// Building.tsx - WRONG
const faction = getFaction('dwarf'); // Always shows dwarf previews!
```

### After (Fixed)
```typescript
// Building.tsx - CORRECT
const faction = getFaction(playerFactionId); // Shows correct faction previews
```

## Implementation Details

### 1. Faction System (`factions.ts`)
- Centralized configuration for all factions
- Each faction has its own units with unique preview images
- Helper functions to safely access faction data

### 2. Building Component (`Building.tsx`)
- Takes `playerFactionId` as a required prop
- Uses player's faction to determine available units
- Displays correct preview images for player's faction
- Spawns units matching the displayed previews

### 3. Selection Panel (`SelectionPanel.tsx`)
- Passes `playerFactionId` to Building component
- Handles both friendly and enemy entity selection
- Prevents showing UI for enemy buildings

## Usage Example

```typescript
import { SelectionPanel } from './ui/SelectionPanel';

function GameUI() {
  const playerFaction = 'human'; // or 'dwarf'
  
  return (
    <SelectionPanel
      selectedEntityId="barracks_001"
      selectedEntityType="building"
      playerFactionId={playerFaction} // Pass player's actual faction
      entities={gameEntities}
      onUnitSpawn={handleUnitSpawn}
    />
  );
}
```

## Testing

To verify the fix:

1. **Test as Human Player**:
   - Select human barracks
   - Verify previews show: Footman, Archer, Knight
   - Train units and verify they match previews

2. **Test as Dwarf Player**:
   - Select dwarf barracks
   - Verify previews show: Warrior, Rifleman, Hammerer
   - Train units and verify they match previews

## Assets Required

The following preview images should be placed in the assets directory:

```
assets/units/human/
  - footman_preview.png
  - archer_preview.png
  - knight_preview.png

assets/units/dwarf/
  - warrior_preview.png
  - rifleman_preview.png
  - hammerer_preview.png
```

## Status

✅ **Bug Fixed** - Human barracks now correctly displays human unit previews.

See `BUG_FIX_DOCUMENTATION.md` for detailed analysis and prevention strategies.
