# ORC-112: Bug vs Fix Comparison

## Visual Demonstration

### Before (BUG) ❌
```
Player selects: DWARF FACTION
Building type: Barracks
Preview shows: 🏰 HUMAN BARRACKS (WRONG!)
                 ↑
                 This is the bug - always shows human buildings
```

### After (FIXED) ✅
```
Player selects: DWARF FACTION
Building type: Barracks
Preview shows: ⛏️ DWARF BARRACKS (CORRECT!)
                 ↑
                 Now correctly shows dwarf buildings
```

## Code Comparison

### building_ghost.gd - The Key Fix

#### ❌ BEFORE (Buggy Code)
```gdscript
## WRONG: This function didn't accept faction parameter
func update_preview(building_type: String):
    # BUG: Always defaulted to human faction!
    var color = FactionConfig.get_building_color(
        FactionConfig.Faction.HUMAN,  # ← Hard-coded to HUMAN!
        building_type
    )
    _update_preview_mesh(building_type, color)
```

#### ✅ AFTER (Fixed Code)
```gdscript
## CORRECT: Now accepts and uses faction parameter
func update_preview(building_type: String, faction: FactionConfig.Faction):
    current_building_type = building_type
    current_faction = faction  # ← Store the faction!
    
    # Get the correct color for THIS faction
    var color = FactionConfig.get_building_color(
        faction,  # ← Use the passed faction parameter!
        building_type
    )
    _update_preview_mesh(building_type, color)
```

## Root Cause Analysis

### Problem
The `BuildingGhost` component's `update_preview()` method:
1. Did NOT accept a `faction` parameter
2. Always defaulted to `FactionConfig.Faction.HUMAN`
3. This caused all building previews to show human faction models

### Solution
1. **Added `faction` parameter** to `update_preview()` method
2. **Store the faction** in `current_faction` member variable
3. **Use the faction** when looking up building colors/models
4. **Ensure callers pass faction** when calling `update_preview()`

## Testing Scenarios

### Test Case 1: Dwarf Faction
```
Given: Player selects Dwarf faction
When: Player enters building placement mode for Barracks
Then: Preview shows DWARF barracks (gray/stone appearance)
```

### Test Case 2: Elf Faction
```
Given: Player selects Elf faction
When: Player enters building placement mode for Town Hall
Then: Preview shows ELF town hall (green/natural appearance)
```

### Test Case 3: Undead Faction
```
Given: Player selects Undead faction
When: Player enters building placement mode for Farm
Then: Preview shows UNDEAD farm (dark/purple appearance)
```

### Test Case 4: Faction Switching
```
Given: Player is in Human faction
When: Player switches to Dwarf faction
And: Building preview is active
Then: Preview immediately updates to show Dwarf building
```

## Impact

### Before Fix
- **User Experience**: Confusing - preview doesn't match what gets built
- **Game Feel**: Unprofessional - broken visual feedback
- **Trust**: Players reported the bug (Gaudio feedback)

### After Fix
- **User Experience**: Clear - preview matches final building
- **Game Feel**: Polished - correct visual feedback
- **Trust**: Players can confidently place buildings

## Files Changed

1. **building_ghost.gd**
   - Added `faction` parameter to `update_preview()`
   - Store faction in `current_faction`
   - Use faction for model/color lookup

2. **main.gd** (caller)
   - Pass `current_faction` when calling `building_ghost.update_preview()`
   - Update preview when faction changes

3. **faction_config.gd** (configuration)
   - Properly defined all faction building models
   - Ensured all faction/building combinations exist

## Verification

✅ Human faction shows human buildings  
✅ Dwarf faction shows dwarf buildings  
✅ Elf faction shows elf buildings  
✅ Undead faction shows undead buildings  
✅ Preview updates when switching factions  
✅ Placed building matches preview  
✅ No console errors or warnings  

## Related User Feedback

> "previews of structures show the human faction equivalent structure (such as, when you are playing as dwarves, and try to create dwarves barracks, the preview/blueprint is showing human barracks)"
> — Gaudio

This issue is now **RESOLVED** ✅
