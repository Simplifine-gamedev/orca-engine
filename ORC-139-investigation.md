# Issue ORC-139 Investigation

## Issue: Minimap shows redundant selected units count

**Linear Issue ID**: ORC-139  
**Type**: Bug  
**Project**: Orca RTS

## Investigation Results

### File Not Found
The issue references `src/ui/Minimap.tsx` for fixing, but this file does not exist in the repository.

### Repository Context
- **Current Repository**: `orca-engine` (Godot engine fork with AI features)
- **Expected Project**: Orca RTS (game project)
- **Searched Paths**: 
  - `/workspace/src/` - Does not exist
  - `/workspace/**/Minimap.tsx` - No results
  - `/workspace/**/*.tsx` - Only 2 files found in `cloud-ide/frontend/app/`

### TypeScript Files Found
1. `/workspace/cloud-ide/frontend/app/layout.tsx` - Cloud IDE layout
2. `/workspace/cloud-ide/frontend/app/page.tsx` - Cloud IDE main page
   - Contains Monaco editor with `minimap: { enabled: false }` (code editor minimap, not RTS game)

### GDScript Search
Searched for GDScript files (`.gd`) that might contain RTS game logic:
- Found only test scripts: `test_script.gd`, `test_error_script.gd`
- No minimap or unit selection logic found

## Possible Explanations

1. **Wrong Repository**: The issue was filed against `orca-engine` instead of a separate `orca-rts` game repository
2. **Missing Game Project**: The RTS game hasn't been created in this repository yet
3. **Incorrect File Path**: The actual file exists in a different location not specified in the issue

## Recommendation

This issue cannot be resolved in the current repository because:
- The file to fix (`src/ui/Minimap.tsx`) does not exist
- No RTS game codebase is present in this repository
- This repository contains the Godot engine fork, not the RTS game built with it

**Action Required**: Verify the correct repository for the Orca RTS game project and re-file the issue there, or clarify if the RTS game should be created in a specific location within this repository.
