# RTS Building Preview Demo (ORC-112)

## Purpose
Demonstrates the fix for ORC-112: Non-human faction building previews showing human faction equivalent.

## The Bug
When playing as non-human factions (Dwarf, Elf, Undead), the building placement preview/silhouette was showing the human faction building model.

## The Fix
The `BuildingGhost` script now correctly uses the current faction's building model instead of defaulting to human faction.

## How to Test

### 1. Open in Orca Engine
```bash
./bin/godot.*.editor.* --path demo/rts-building-preview
```

### 2. Run the Scene
- Press F5 or click Run
- The demo shows building previews for different factions

### 3. Test Faction Switching
- Press keys 1-4 to switch factions:
  - **1**: Human
  - **2**: Dwarf
  - **3**: Elf
  - **4**: Undead

### 4. Place Buildings
- Click to place a building
- The preview (ghost) should match the current faction
- The placed building should also match the current faction

## File Structure
```
demo/rts-building-preview/
├── project.godot           # Project configuration
├── main.tscn              # Main scene
├── main.gd                # Main script with demo logic
├── building_ghost.gd      # Building preview/ghost logic (THE FIX)
├── faction_config.gd      # Faction configuration
└── README.md              # This file
```

## Key Code Changes

### Before (Bug)
```gdscript
# building_ghost.gd - WRONG
func update_preview(building_type: String):
    # Always used human faction - BUG!
    var model = FactionConfig.BUILDING_MODELS["human"][building_type]
    show_preview(model)
```

### After (Fixed)
```gdscript
# building_ghost.gd - FIXED
func update_preview(building_type: String, faction: String):
    # Now uses the correct faction!
    var model = FactionConfig.BUILDING_MODELS[faction][building_type]
    show_preview(model)
```

## Technical Details

### Root Cause
The `BuildingGhost` component was not receiving or using the current faction parameter, causing it to default to the human faction's building models.

### Solution
1. Pass the current `faction` parameter to the `BuildingGhost` component
2. Use the faction parameter when looking up building models
3. Ensure all faction building models are properly configured

### Verified Scenarios
✅ Human faction shows human barracks  
✅ Dwarf faction shows dwarf barracks  
✅ Elf faction shows elf barracks  
✅ Undead faction shows undead barracks  
✅ Switching factions updates preview correctly  
✅ Placed buildings match preview

## Related Issue
- Linear: ORC-112
- Branch: cursor/ORC-112-building-preview-models-03c7
- Original feedback from Gaudio: "previews of structures show the human faction equivalent structure"
