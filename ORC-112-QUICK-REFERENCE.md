# ORC-112 Quick Reference

## The Bug (One Line)
Building previews always showed human faction models instead of the selected faction's models.

## The Fix (One Line)
Pass the `faction` parameter to `BuildingGhost.update_preview()` and use it for model lookup.

## Before/After Code

### Before ❌
```gdscript
func update_preview(building_type: String):
    var color = FactionConfig.get_building_color(
        FactionConfig.Faction.HUMAN,  # Hard-coded!
        building_type
    )
```

### After ✅
```gdscript
func update_preview(building_type: String, faction: FactionConfig.Faction):
    var color = FactionConfig.get_building_color(
        faction,  # Uses provided faction!
        building_type
    )
```

## How to Test
```bash
./bin/godot.*.editor.* --path demo/rts-building-preview
# Press 1-4 to switch factions
# Verify preview changes color/style
```

## Files Changed
- `building_ghost.gd` - Added faction parameter
- `main.gd` - Pass faction when calling update_preview()
- `faction_config.gd` - Centralized faction data

## Verification Checklist
- [x] Human faction shows human buildings (brown)
- [x] Dwarf faction shows dwarf buildings (gray)
- [x] Elf faction shows elf buildings (green)
- [x] Undead faction shows undead buildings (purple)
- [x] Preview updates on faction switch
- [x] Placed building matches preview

## Documentation
- **Investigation**: ORC-112-INVESTIGATION.md
- **Solution**: ORC-112-SOLUTION.md
- **Testing**: demo/rts-building-preview/TESTING_GUIDE.md
- **Architecture**: demo/rts-building-preview/ARCHITECTURE.md
- **Project Summary**: ORC-112-PROJECT-SUMMARY.md
- **Linear Summary**: ORC-112-LINEAR-SUMMARY.md

## Status
✅ **RESOLVED** - Ready for review and merge

## Branch
`cursor/ORC-112-building-preview-models-03c7`

## Commits
7 commits total, including investigation, implementation, and documentation.

## Key Insight
**Explicit over Implicit**: Always pass required context (faction) as a parameter rather than relying on defaults or global state.

---

For full details, see **ORC-112-PROJECT-SUMMARY.md**
