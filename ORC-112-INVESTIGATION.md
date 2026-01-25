# ORC-112 Investigation Report

## Issue Summary
**Title**: [Bug] Non-human faction building previews show human faction equivalent  
**Project**: Orca RTS  
**Branch**: cursor/ORC-112-building-preview-models-03c7

## Problem
When playing as non-human factions (Dwarf, Elf, Undead, etc.), the building placement preview/silhouette shows the human faction building model instead of the correct faction model.

## Investigation Findings

### Repository Mismatch
The Linear issue references the following files that **do not exist** in this repository:
- `src/buildings/Building.tsx` (BuildingGhost component)
- `src/buildings/buildingModels.ts`
- `src/config/factions.ts`

### Current Repository
This is the **Orca Engine** repository (Godot engine fork), not an RTS game project. The engine contains:
- C++ engine code
- Godot editor modifications
- Cloud IDE infrastructure
- No RTS game implementation

### Referenced Commit Not Found
The issue mentions commit `9a2f72d` with note "preview still messsed up" - this commit does not exist in this repository's history.

### Organization Repositories
Checked Simplifine-gamedev GitHub organization. Found only 3 repositories:
1. orca-engine (this repo)
2. docs
3. Simplifine

No separate "Orca RTS" repository exists.

## Possible Scenarios

### Scenario 1: Wrong Repository
The Linear issue was created for the wrong repository. The RTS game may exist elsewhere or is a private project.

### Scenario 2: Missing Game Code
The RTS game code should exist but hasn't been committed or is in a different branch.

### Scenario 3: Future Development
The issue was created prematurely for a game project that hasn't been started yet.

## Recommended Actions

### Option A: Clarify Issue Location
- Confirm which repository contains the RTS game code
- Update Linear issue with correct file paths
- Transfer issue to correct repository if needed

### Option B: Create Demo RTS Game
If the RTS game should be part of this repository, we could create:
1. A `demo/rts-game/` directory with React/TypeScript implementation
2. Implement faction system with building previews
3. Fix the preview bug as described

### Option C: Godot Implementation
Since this is a Godot engine repository, implement the RTS game as a Godot project:
1. Create `demo/rts/` with GDScript implementation
2. Implement faction-specific building previews
3. Fix the preview rendering logic

## Technical Analysis (If Implemented)

The bug description suggests the issue is in the building preview/ghost rendering logic. Typical causes:

### 1. Hard-coded Model References
```typescript
// ❌ Wrong: Hard-coded to human faction
const previewModel = buildingModels.human[buildingType];

// ✅ Correct: Use current faction
const previewModel = buildingModels[currentFaction][buildingType];
```

### 2. Missing Faction Context
```typescript
// BuildingGhost component might not receive faction prop
<BuildingGhost 
  buildingType={type}
  faction={currentFaction}  // ← This might be missing
  position={position}
/>
```

### 3. Model Loading Logic
```typescript
// buildingModels.ts might default to human faction
export const getBuildingPreview = (type: string, faction?: string) => {
  const factionToUse = faction || 'human';  // ← Default causes bug
  return models[factionToUse][type];
};
```

## Next Steps

**Awaiting clarification on:**
1. Where is the actual RTS game code located?
2. Should it be part of this repository?
3. Is this a React/TypeScript project or Godot project?

Once clarified, I can proceed with fixing the building preview bug.

---

**Investigation Date**: January 25, 2026  
**Investigated By**: Cloud Agent  
**Branch**: cursor/ORC-112-building-preview-models-03c7
