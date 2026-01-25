# ORC-200 Investigation Report

## Issue: Dwarf Building Previews Broken

### Summary
After thorough investigation, the files mentioned in Linear issue ORC-200 do not exist in the `orca-engine` repository.

### Files Referenced in Issue (NOT FOUND)
- `src/buildings/buildingModels.ts`
- `src/config/factions.ts`  
- `generated_factions/all_factions_buildings.json`

### Investigation Details

#### Repository Analysis
- **Current Repo**: `orca-engine` (Godot Engine fork with AI integration)
- **Expected**: RTS game project with building/faction systems
- **Finding**: No RTS game code exists in this repository

#### Search Results
1. No TypeScript files related to buildings or factions
2. No JSON files containing faction data
3. No references to "dwarf", "faction", or building preview systems
4. No RTS-specific game logic found

#### Organization Repositories
Searched all repos in Simplifine-gamedev organization:
- `orca-engine` - Game engine (current repo)
- `docs` - Documentation
- `Simplifine` - LLM finetuning tool

**Result**: No separate "Orca RTS" repository found.

### Conclusion

This Linear issue appears to be:
1. **Created for the wrong repository**, OR
2. **References a game project that doesn't exist yet**, OR  
3. **Linear integration is misconfigured**

The "Orca RTS" project mentioned in the issue does not exist in this codebase. The building preview system, dwarf faction configuration, and related files would need to be located in a separate game project repository.

### Recommendation

1. Verify the correct repository for the "Orca RTS" project
2. Check if the RTS game is maintained in a private repository
3. Update Linear integration to point to the correct repository
4. If this is a new project, clarify requirements for creating the RTS game structure

---

**Investigated by**: Cursor AI Agent  
**Date**: January 25, 2026  
**Branch**: `cursor/ORC-200-dwarf-building-previews-988b`
