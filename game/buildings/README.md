# Wall Building System

This directory contains the improved wall building system for Orca RTS.

## Files

- `WallSystem.gd` - Core wall building logic with UX improvements
- `WallSystem.tscn` - Scene file for the wall system
- `Wall.gd` - Individual wall instance (to be created)
- `Wall.tscn` - Wall entity scene (to be created)

## Features & UX Improvements

### 1. Right-Click to Cancel
- Users can now right-click to cancel wall placement (more intuitive than ESC)
- ESC key still works as a backup option
- Resolves the confusion reported in user feedback

### 2. Visual Feedback During Placement
- Green overlay for valid placement areas
- Red overlay for invalid placement areas
- Grid-based highlighting around cursor
- Preview of wall before placing

### 3. Cost Preview
- Cost is displayed before confirming placement
- Color-coded based on affordability:
  - Green tint: Can afford
  - Red tint: Cannot afford
- Real-time resource tracking

### 4. Tutorial Tooltip
- Shows automatically on first wall build
- Explains controls and mechanics
- Can be dismissed or auto-hides after 5 seconds
- Remembers if user has seen it (saved to settings)

### 5. Valid Placement Area Highlights
- Highlights valid grid cells around cursor
- Shows placement grid for precise building
- Snaps to grid for clean alignment
- Real-time collision detection

## Usage

### Basic Setup

```gdscript
# In your main game scene
extends Node2D

@onready var wall_system = $WallSystem
@onready var ui_panel = $UI/WallBuildPanel

func _ready():
    # Connect wall system to UI
    wall_system.set_ui_panel(ui_panel)
    
    # Connect signals
    wall_system.wall_placed.connect(_on_wall_placed)
    wall_system.wall_cancelled.connect(_on_wall_cancelled)
    
    ui_panel.build_requested.connect(_on_build_requested)
    ui_panel.cancel_requested.connect(_on_cancel_requested)
    
    # Set initial resources
    ui_panel.set_player_resources(500)

func _on_build_requested():
    if wall_system.can_afford_wall(player_resources):
        wall_system.start_placement()
    else:
        ui_panel.show_error("Not enough resources!")

func _on_cancel_requested():
    wall_system.cancel_placement()

func _on_wall_placed(position: Vector2):
    # Deduct resources
    player_resources -= wall_system.get_placement_cost()
    ui_panel.set_player_resources(player_resources)
    ui_panel.show_success("Wall placed!")

func _on_wall_cancelled():
    ui_panel.show_status("Wall placement cancelled")
```

### Advanced Configuration

```gdscript
# Customize wall properties
wall_system.wall_cost = 75
wall_system.wall_health = 1000
wall_system.grid_size = 64
wall_system.placement_color_valid = Color(0, 1, 0, 0.6)
wall_system.placement_color_invalid = Color(1, 0, 0, 0.6)
```

## Controls

- **Left Click**: Place wall at cursor position
- **Right Click**: Cancel wall placement (primary method)
- **ESC**: Cancel wall placement (backup method)
- **Mouse Move**: Preview placement and see valid areas

## Implementation Details

### Collision Detection

The system uses an `Area2D` to detect collisions with existing structures:
- Collision layer 0 (doesn't collide)
- Collision mask 1 (detects buildings on layer 1)

### Grid Snapping

Walls snap to a configurable grid (default 32 pixels):
```gdscript
func snap_to_grid(pos: Vector2) -> Vector2:
    return Vector2(
        floor(pos.x / grid_size) * grid_size,
        floor(pos.y / grid_size) * grid_size
    )
```

### Validation

Placement is validated against:
1. Map boundaries
2. Existing structures (collision detection)
3. Terrain type (water, cliffs, etc.)
4. Resource availability

## Signals

### WallSystem

- `wall_placed(position: Vector2)` - Emitted when wall is successfully placed
- `wall_cancelled()` - Emitted when placement is cancelled
- `placement_mode_changed(active: bool)` - Emitted when entering/exiting placement mode

### WallBuildPanel

- `build_requested()` - Emitted when build button is clicked
- `cancel_requested()` - Emitted when cancel button is clicked

## Future Enhancements

- [ ] Wall chaining (place multiple connected walls)
- [ ] Wall upgrade system
- [ ] Different wall types (stone, wood, metal)
- [ ] Wall health indicators
- [ ] Repair functionality
- [ ] Auto-wall formation around buildings

## Testing

To test the wall building system:

1. Open the Orca Engine editor
2. Load the `WallSystemExample.tscn` scene
3. Run the scene
4. Click "Build Wall" button
5. Move mouse to see valid placement areas
6. Left-click to place, right-click to cancel

## User Feedback Addressed

✅ "Press escape to cancel (walls) is confusing them"
- **Solution**: Right-click is now the primary cancel method

✅ "Wall building is not super intuitive"
- **Solution**: Added visual feedback, cost preview, tutorial tooltip, and placement highlights

## Credits

Created for Orca RTS engine based on user feedback from the community.
