# RTS Unit Spawning Fix - ORC-118

## Problem

When a building has no rally point set, spawned units appear **inside the building** and are hard to see/select.

## User Feedback

* "without the rally point, units get spawned inside the building - he didnt see the spawned units"
* "Worker got spawned randomly inside the building"

## Solution

This example demonstrates the proper way to handle unit spawning in RTS games to ensure units always spawn at a visible, accessible location outside the building.

### Key Implementation Details

**Building.gd - get_spawn_position()**

```gdscript
func get_spawn_position() -> Vector2:
    """
    Get the position where units should spawn.
    Returns a position outside the building to avoid units spawning inside.
    """
    if has_rally_point:
        # Spawn towards rally point
        var direction_to_rally := (rally_point - global_position).normalized()
        return global_position + direction_to_rally * default_spawn_offset.length()
    else:
        # DEFAULT: Spawn at fixed offset from building (FIXES THE BUG)
        return global_position + default_spawn_offset
```

### Before vs After

**Before (Bug):**
- Unit spawns at `building.global_position` (inside the building)
- Unit is hidden and hard to select
- Player doesn't see the spawned unit

**After (Fixed):**
- Unit spawns at `building.global_position + default_spawn_offset`
- Unit appears outside the building entrance
- Unit is visible and selectable immediately

## Usage

### 1. Basic Building Setup

```gdscript
var building = RTSBuilding.new()
building.spawn_scene = preload("res://Unit.tscn")
building.default_spawn_offset = Vector2(100, 0)  # 100 pixels to the right
building.spawn_interval = 2.0
add_child(building)
```

### 2. Start Production

```gdscript
building.start_production()
# Units will now spawn every 2 seconds outside the building
```

### 3. Optional: Set Rally Point

```gdscript
# Units will move to this position after spawning
building.set_rally_point(Vector2(500, 300))
```

### 4. Optional: Clear Rally Point

```gdscript
# Units will spawn at default offset and stay there
building.clear_rally_point()
```

## Configuration Options

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `spawn_scene` | PackedScene | null | The unit scene to instantiate |
| `default_spawn_offset` | Vector2 | (100, 0) | Offset from building center for spawn position |
| `spawn_interval` | float | 2.0 | Seconds between spawns |
| `rally_point` | Vector2 | (0, 0) | World position for rally point |
| `has_rally_point` | bool | false | Whether rally point is active |

## Integration with Existing Games

If you have existing building/spawning code:

1. **Add default_spawn_offset property:**
   ```gdscript
   @export var default_spawn_offset := Vector2(100, 0)
   ```

2. **Update spawn position calculation:**
   ```gdscript
   # OLD (buggy):
   unit.global_position = building.global_position
   
   # NEW (fixed):
   unit.global_position = building.global_position + default_spawn_offset
   ```

3. **Handle rally points properly:**
   ```gdscript
   if has_rally_point:
       # Spawn towards rally point direction
       var direction = (rally_point - building.global_position).normalized()
       spawn_pos = building.global_position + direction * default_spawn_offset.length()
   ```

## Testing

Run the example scene to see:
- Units spawning outside the building (green circle shows spawn point)
- Rally point system (yellow circle and line)
- Unit movement to rally point

Debug visualization shows:
- **Blue rectangle**: Building bounds
- **Green circle**: Spawn position
- **Yellow circle**: Rally point (if set)
- **Red circle**: Unit

## Technical Notes

### Why This Fix Works

1. **Consistent spawn location**: Units always spawn at a predictable offset
2. **Visibility**: Spawn position is outside building collision/visual bounds
3. **Rally point compatibility**: System works with or without rally points
4. **Easy configuration**: Designers can adjust `default_spawn_offset` per building type

### Building Type Examples

```gdscript
# Barracks (horizontal exit)
barracks.default_spawn_offset = Vector2(80, 0)

# Factory (bottom exit)
factory.default_spawn_offset = Vector2(0, 100)

# Starport (centered above)
starport.default_spawn_offset = Vector2(0, -120)
```

## See Also

- Godot MultiplayerSpawner documentation
- RTS game design patterns
- Unit selection and pathfinding systems
