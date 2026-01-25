# Building Preview Architecture

## Component Flow Diagram

### Before Fix (Buggy) ❌

```
┌─────────────────────────────────────────────────────────┐
│ Player Action: Select Dwarf Faction                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Main.gd                                                  │
│  current_faction = FactionConfig.Faction.DWARF          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ ❌ BUG: Doesn't pass faction!
                  │ building_ghost.update_preview(building_type)
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ BuildingGhost.gd (BUGGY)                                │
│  func update_preview(building_type: String):            │
│    # ❌ No faction parameter!                           │
│    # ❌ Defaults to human faction                       │
│    color = FactionConfig.get_building_color(            │
│        FactionConfig.Faction.HUMAN,  # ❌ HARD-CODED!  │
│        building_type                                     │
│    )                                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Result: Shows HUMAN building preview                    │
│ Expected: Dwarf building preview                        │
│ ❌ BUG: Wrong faction shown!                            │
└─────────────────────────────────────────────────────────┘
```

### After Fix (Correct) ✅

```
┌─────────────────────────────────────────────────────────┐
│ Player Action: Select Dwarf Faction                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Main.gd                                                  │
│  current_faction = FactionConfig.Faction.DWARF          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ ✅ FIX: Pass faction parameter!
                  │ building_ghost.update_preview(building_type, current_faction)
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ BuildingGhost.gd (FIXED)                                │
│  func update_preview(building_type: String,             │
│                       faction: FactionConfig.Faction):  │
│    # ✅ Accepts faction parameter                       │
│    current_faction = faction  # ✅ Store it             │
│    color = FactionConfig.get_building_color(            │
│        faction,  # ✅ Use provided faction!             │
│        building_type                                     │
│    )                                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Result: Shows DWARF building preview                    │
│ Expected: Dwarf building preview                        │
│ ✅ CORRECT: Faction matches!                            │
└─────────────────────────────────────────────────────────┘
```

## State Flow

### Faction Selection Flow

```
User Input          Main State           BuildingGhost State
─────────────────   ──────────────────   ────────────────────

Press '1' (Human)
     │
     ├──────────► current_faction        
     │              = HUMAN              
     │                   │               
     │                   └──────────────► current_faction = HUMAN
     │                                    color = human color
     │                                    mesh = human style
     │
Press '2' (Dwarf)
     │
     ├──────────► current_faction        
     │              = DWARF              
     │                   │               
     │                   └──────────────► current_faction = DWARF
     │                                    color = dwarf color  ✅
     │                                    mesh = dwarf style  ✅
     │
Press '3' (Elf)
     │
     ├──────────► current_faction        
     │              = ELF                
     │                   │               
     │                   └──────────────► current_faction = ELF
     │                                    color = elf color    ✅
     │                                    mesh = elf style    ✅
```

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ FactionConfig (faction_config.gd)                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ Single Source of Truth for Faction Data                     │
│                                                              │
│ BUILDING_MODELS = {                                          │
│   "human": { "barracks": "...", "town_hall": "..." },       │
│   "dwarf": { "barracks": "...", "town_hall": "..." },       │
│   "elf": { "barracks": "...", "town_hall": "..." },         │
│   "undead": { "barracks": "...", "town_hall": "..." }       │
│ }                                                            │
│                                                              │
│ BUILDING_COLORS = {                                          │
│   "human": { "barracks": Color(...), ... },                 │
│   "dwarf": { "barracks": Color(...), ... },                 │
│   ...                                                        │
│ }                                                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ get_building_color(faction, type)
                         │ get_building_model(faction, type)
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────────┐             ┌──────────────────┐
│ Main.gd          │             │ BuildingGhost.gd │
│                  │   update    │                  │
│ Stores:          │─────────────►│ Receives:        │
│ - current_faction│   _preview( │ - building_type  │
│ - current_type   │   type,     │ - faction ✅     │
│                  │   faction)  │                  │
│ Controls:        │             │ Renders:         │
│ - Faction switch │             │ - Preview mesh   │
│ - Building cycle │             │ - Faction color  │
│ - Placement      │             │ - Transparency   │
└──────────────────┘             └──────────────────┘
```

## Component Responsibilities

### FactionConfig.gd
**Purpose**: Centralized faction data
- ✅ Define all factions
- ✅ Define building models per faction
- ✅ Define building colors per faction
- ✅ Provide lookup functions

**Key Functions**:
```gdscript
get_faction_key(faction: Faction) -> String
get_building_color(faction: Faction, type: String) -> Color
get_building_model(faction: Faction, type: String) -> String
```

### BuildingGhost.gd
**Purpose**: Render building preview
- ✅ Accept faction parameter (THE FIX)
- ✅ Store current faction
- ✅ Look up faction-specific model/color
- ✅ Render semi-transparent preview
- ✅ Show valid/invalid placement

**Key Functions**:
```gdscript
update_preview(building_type: String, faction: Faction)  # ✅ Fixed signature
set_valid_placement(valid: bool)
show_preview()
hide_preview()
```

### Main.gd
**Purpose**: Game logic and user input
- ✅ Track current faction
- ✅ Handle faction switching
- ✅ Pass faction to BuildingGhost (THE FIX)
- ✅ Handle building placement
- ✅ Update UI

**Key Functions**:
```gdscript
_switch_faction(new_faction: Faction)
_cycle_building_type()
_place_building()
_update_ghost_position()
```

## Parameter Passing Chain

The key to the fix is ensuring the faction flows through the entire chain:

```
User Input (Press '2' for Dwarf)
    │
    ▼
_input() detects KEY_2
    │
    ▼
_switch_faction(FactionConfig.Faction.DWARF)
    │
    ├─► current_faction = DWARF  (Store in Main)
    │
    └─► building_ghost.update_preview(
            current_building_type,
            current_faction  # ✅ Pass it!
        )
            │
            ▼
        BuildingGhost receives faction
            │
            ├─► current_faction = faction  (Store in Ghost)
            │
            └─► color = FactionConfig.get_building_color(
                    faction,  # ✅ Use it!
                    building_type
                )
                    │
                    ▼
                Render preview with faction-specific color/model
```

## The Fix in Different Languages

### GDScript (Godot) - Our Implementation
```gdscript
# ✅ Fixed
func update_preview(building_type: String, faction: FactionConfig.Faction):
    var color = FactionConfig.get_building_color(faction, building_type)
```

### TypeScript (React)
```typescript
// ✅ Fixed
function BuildingGhost({ buildingType, faction }: Props) {
  const model = buildingModels[faction][buildingType];
  return <PreviewMesh model={model} />;
}
```

### C# (Unity)
```csharp
// ✅ Fixed
public void UpdatePreview(string buildingType, Faction faction) {
    var model = FactionConfig.GetBuildingModel(faction, buildingType);
    ShowPreview(model);
}
```

### Python (Generic)
```python
# ✅ Fixed
def update_preview(self, building_type: str, faction: Faction):
    color = FactionConfig.get_building_color(faction, building_type)
    self.render_preview(color)
```

## Key Design Principles

### 1. Explicit Parameter Passing
❌ Don't rely on defaults or implicit state
✅ Pass faction explicitly as a parameter

### 2. Single Source of Truth
❌ Don't duplicate faction data
✅ Centralize in FactionConfig

### 3. Immediate Propagation
❌ Don't batch or delay faction changes
✅ Update preview immediately when faction changes

### 4. Type Safety
❌ Don't use strings for factions
✅ Use enum/type for compile-time safety

### 5. Separation of Concerns
❌ Don't mix game logic with rendering
✅ Main handles logic, BuildingGhost handles rendering

## Testing Strategy

### Unit Tests (Conceptual)
```gdscript
func test_preview_uses_faction():
    var ghost = BuildingGhost.new()
    
    # Test human faction
    ghost.update_preview("barracks", FactionConfig.Faction.HUMAN)
    assert(ghost.current_faction == FactionConfig.Faction.HUMAN)
    
    # Test dwarf faction
    ghost.update_preview("barracks", FactionConfig.Faction.DWARF)
    assert(ghost.current_faction == FactionConfig.Faction.DWARF)
    
    # Test that different factions produce different colors
    var human_color = FactionConfig.get_building_color(
        FactionConfig.Faction.HUMAN, "barracks"
    )
    var dwarf_color = FactionConfig.get_building_color(
        FactionConfig.Faction.DWARF, "barracks"
    )
    assert(human_color != dwarf_color)
```

### Integration Tests (Interactive Demo)
1. Faction switching updates preview
2. All building types work for all factions
3. Placed buildings match previews
4. No errors in console

## Performance Considerations

### Why This Fix is Efficient
1. **No Additional Allocations**: Just passing an enum value
2. **Immediate Update**: No async operations needed
3. **Cached Lookups**: FactionConfig uses dictionaries for O(1) lookup
4. **No Duplication**: Single faction storage, not per-building

### Optimization Opportunities
```gdscript
# Cache the color for current faction + building type
var _cached_color: Color
var _cached_faction: Faction
var _cached_type: String

func update_preview(building_type: String, faction: Faction):
    # Only recompute if faction or type changed
    if faction != _cached_faction or building_type != _cached_type:
        _cached_color = FactionConfig.get_building_color(faction, building_type)
        _cached_faction = faction
        _cached_type = building_type
    
    _update_preview_mesh(building_type, _cached_color)
```

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Default Parameters
```gdscript
# DON'T DO THIS
func update_preview(building_type: String, faction = null):
    if faction == null:
        faction = FactionConfig.Faction.HUMAN  # ❌ Hidden default!
```

### ❌ Pitfall 2: Global State
```gdscript
# DON'T DO THIS
var global_current_faction = FactionConfig.Faction.HUMAN

func update_preview(building_type: String):
    var color = FactionConfig.get_building_color(
        global_current_faction,  # ❌ Hidden dependency!
        building_type
    )
```

### ❌ Pitfall 3: Late Binding
```gdscript
# DON'T DO THIS
func update_preview(building_type: String):
    # ❌ Assuming faction will be set later
    var color = FactionConfig.get_building_color(
        current_faction,  # ❌ Might not be initialized!
        building_type
    )
```

### ✅ Best Practice: Explicit, Required Parameter
```gdscript
# DO THIS
func update_preview(building_type: String, faction: FactionConfig.Faction):
    # ✅ Faction is required, no defaults, no ambiguity
    var color = FactionConfig.get_building_color(faction, building_type)
```

## Conclusion

The architecture fix is simple but critical:
1. **Add faction parameter** to update_preview()
2. **Pass faction explicitly** from caller
3. **Use faction consistently** for all lookups
4. **No defaults or implicit behavior**

This ensures building previews always match the player's selected faction.
