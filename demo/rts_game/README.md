# RTS Game - Resource Pooling Clarity Improvements

This demo addresses Linear issue **ORC-153** about resource pooling clarity for players.

## Overview

This RTS game demo implements a comprehensive resource management system with clear UI/UX improvements to help players understand:
- What resources they have
- How resources are generated
- What things cost
- Whether they can afford something
- What each resource is used for

## Files Structure

```
demo/rts_game/
├── ui/
│   ├── ResourceBar.gd              # Main resource display with tooltips
│   ├── WorkerBuildPanel.gd         # Unit training with cost display
│   └── BuildingPlacementPanel.gd   # Building placement with costs
├── buildings/
│   ├── Building.gd                 # Base building class
│   ├── GoldMine.gd                 # Gold-generating building
│   ├── Barracks.gd                 # Military unit training
│   └── Farm.gd                     # Food-generating building
└── README.md                       # This file
```

## Features Implemented

### 1. ✅ Tutorial/Tooltips Explaining Resources

**Location:** `ui/ResourceBar.gd`

- **First-time tutorial**: Popup dialog on first launch explaining all resources
- **Hover tooltips**: Rich tooltips on each resource showing:
  - Resource name and description
  - Current amount
  - Income rate (+X/sec)
  - Usage tips
- **Persistent tutorial completion**: Saves to user data to not show again

**Example:**
```gdscript
💰 Gold
Gold - Used for buildings and units
Income: +5 per second
Tip: Build more gatherers to increase income!
```

### 2. ✅ Resource Costs on Building/Unit Buttons

**Locations:** 
- `ui/WorkerBuildPanel.gd` - Unit training costs
- `ui/BuildingPlacementPanel.gd` - Building costs
- `buildings/Building.gd` - Individual building cost display

**Features:**
- Costs displayed directly on buttons with icons
- Detailed tooltips showing:
  - Full cost breakdown
  - Build time
  - Unit/building description
- Real-time cost comparison with current resources

**Example button text:**
```
⚔️ Soldier
Basic combat unit
💰100 | 🌾25
```

### 3. ✅ Highlight When Can't Afford Something

**Locations:**
- `ui/ResourceBar.gd` - Red flashing on insufficient resources
- `ui/WorkerBuildPanel.gd` - Grayed out unaffordable units
- `ui/BuildingPlacementPanel.gd` - Grayed out unaffordable buildings

**Features:**
- **Real-time affordability checking**: Buttons update every frame
- **Visual feedback**:
  - Grayed out buttons (50% opacity) when unaffordable
  - Red flashing resource amounts when attempting to buy
- **Detailed shortage messages**: Popup dialogs showing exactly what's needed
- **Missing amount indicators**: Shows "need X more" for each resource

**Example insufficient resources dialog:**
```
Insufficient Resources!

To train ⚔️ Soldier, you need:
💰 Gold: need 50 more
🌾 Food: need 15 more

Wait for resource income or build more gatherers!
```

### 4. ✅ Resource Income Indicators (+X/sec)

**Location:** `ui/ResourceBar.gd`

**Features:**
- **Income display**: Green "+X/sec" label under each resource
- **Automatic resource generation**: Timer-based income every second
- **Visual income tracking**: Players can see resources incrementing
- **Building-based income**: Resource-generating buildings add to income

**Example:**
```
💰 500
+5/sec  (in green)
```

### 5. ✅ Show What Each Resource Is Used For

**Locations:**
- `ui/ResourceBar.gd` - Resource descriptions in tooltips
- `buildings/Building.gd` - Detailed building info
- `ui/WorkerBuildPanel.gd` - Unit descriptions

**Features:**
- **Resource descriptions**: Each resource has a clear description:
  - 💰 Gold: "Used for buildings and units"
  - 🪵 Wood: "Used for construction"
  - 🌾 Food: "Needed to support units"
  - 🪨 Stone: "Required for advanced buildings"
- **Usage examples**: Costs shown on all buildings and units
- **Contextual help**: Tooltips explain what requires each resource

## Resource System

### Resources Available
1. **💰 Gold** - Primary currency for buildings and units
2. **🪵 Wood** - Construction material
3. **🌾 Food** - Unit upkeep and training
4. **🪨 Stone** - Advanced buildings

### Units Available
- **👷 Worker** - Gathers resources (💰50, 🌾10)
- **⚔️ Soldier** - Basic combat unit (💰100, 🌾25)
- **🏹 Archer** - Ranged unit (💰80, 🪵30, 🌾20)
- **🐎 Cavalry** - Fast mounted unit (💰150, 🌾40)

### Buildings Available
- **⛏️ Gold Mine** - Generates +10 gold/sec (💰150, 🪵100, 🪨50)
- **🏰 Barracks** - Train military units (💰200, 🪵150, 🪨100)
- **🌾 Farm** - Produces +5 food/sec (💰80, 🪵60)
- **🪓 Lumber Mill** - Generates wood (💰100, 🪨30)
- **🪨 Quarry** - Produces stone (💰120, 🪵80)

## User Experience Improvements

### Problem: "Players don't understand the resource system"

**Solutions Implemented:**

1. **Visual Clarity**
   - Large, clear resource icons with emojis
   - Color-coded feedback (green for income, red for insufficient)
   - Progress bars for construction and health

2. **Information Hierarchy**
   - Most important info (costs) on buttons
   - Detailed info in tooltips
   - Contextual help when actions fail

3. **Immediate Feedback**
   - Real-time affordability updates
   - Red flashing when can't afford
   - Clear error messages explaining what's needed

4. **Progressive Disclosure**
   - Tutorial on first launch
   - Tooltips for deeper information
   - Detailed info dialogs when needed

5. **Predictive UI**
   - Show income rates so players can plan
   - Display "need X more" instead of just disabling
   - Build queues show what's being produced

## Usage Example

```gdscript
# In your main game scene:

extends Node2D

@onready var resource_bar = $ResourceBar
@onready var worker_build_panel = $WorkerBuildPanel
@onready var building_panel = $BuildingPlacementPanel

func _ready():
	# Connect panels to resource bar
	worker_build_panel.resource_bar = resource_bar
	building_panel.resource_bar = resource_bar
	
	# Connect signals
	worker_build_panel.unit_training_complete.connect(_on_unit_complete)
	building_panel.building_placed.connect(_on_building_placed)

func _on_unit_complete(unit_type):
	print("Training complete: ", unit_type)

func _on_building_placed(building_type, position):
	print("Building placed: ", building_type, " at ", position)
```

## Testing Checklist

- [x] Resources display with icons and amounts
- [x] Income indicators show +X/sec
- [x] Tooltips appear on hover
- [x] Tutorial shows on first launch
- [x] Buttons gray out when unaffordable
- [x] Red flashing on insufficient resources
- [x] Detailed shortage messages
- [x] Cost display on all buttons
- [x] Build queue shows progress
- [x] Buildings show construction progress
- [x] Resource-generating buildings work

## Technical Details

### Architecture
- **Component-based design**: Each UI element is self-contained
- **Signal-driven**: Loose coupling between components
- **Resource pooling**: Centralized resource management in ResourceBar
- **Affordability checking**: Real-time validation before spending

### Performance
- **Efficient updates**: Only update UI when resources change
- **Minimal polling**: Process() only for affordability checks
- **Smart caching**: Resource amounts cached in dictionaries

### Extensibility
- Easy to add new resources (just add to resources dictionary)
- Easy to add new units/buildings (inherit from Building class)
- Modular UI components can be reused

## Future Enhancements

Potential additions (not required for ORC-153):
- Resource storage caps
- Trade system between resources
- Technology/upgrades that reduce costs
- Efficiency bonuses for multiple resource buildings
- Visual resource flow animations
- Sound effects for resource collection
- Resource shortage warnings before reaching zero

## Credits

Implemented by Cursor AI Agent for Linear issue ORC-153
Game: Orca RTS
Engine: Orca Engine (Godot-based)
