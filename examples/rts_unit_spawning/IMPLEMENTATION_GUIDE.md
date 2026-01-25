# Implementation Guide: Fixing Unit Spawn Location in RTS Games

## Issue: ORC-118 - Units Spawn Inside Building

### Problem Statement

When buildings spawn units without a rally point set, units appear inside the building geometry, making them:
- Invisible to the player
- Hard to select
- Potentially stuck in collision geometry
- Creating a poor user experience

### Root Cause

The bug occurs when spawn position is set directly to the building's position:

```gdscript
# BUGGY CODE:
var unit = spawn_scene.instantiate()
unit.global_position = building.global_position  # ❌ This is inside the building!
```

### Solution: Default Spawn Offset

Always spawn units at an offset from the building center:

```gdscript
# FIXED CODE:
var unit = spawn_scene.instantiate()
unit.global_position = building.global_position + default_spawn_offset  # ✅ Outside!
```

## Implementation Steps

### Step 1: Add Spawn Offset Property

```gdscript
class_name Building extends Node2D

@export var default_spawn_offset := Vector2(100, 0)  # Customize per building type
```

### Step 2: Create Spawn Position Calculator

```gdscript
func get_spawn_position() -> Vector2:
    if has_rally_point:
        # Spawn towards rally point
        var direction = (rally_point - global_position).normalized()
        return global_position + direction * default_spawn_offset.length()
    else:
        # DEFAULT: Use fixed offset (FIXES THE BUG)
        return global_position + default_spawn_offset
```

### Step 3: Use in Spawn Function

```gdscript
func spawn_unit() -> Node:
    var unit = spawn_scene.instantiate()
    get_parent().add_child(unit)
    
    # Apply the fix
    unit.global_position = get_spawn_position()  # ✅ Always outside building
    
    # Optional: Send to rally point
    if has_rally_point and unit.has_method("move_to"):
        unit.move_to(rally_point)
    
    return unit
```

## TypeScript/JavaScript Implementation

If your game uses TypeScript (like the mentioned `gameStore.ts`):

```typescript
// Building.ts
export class Building {
    position: Vector2;
    defaultSpawnOffset: Vector2 = { x: 100, y: 0 };
    rallyPoint?: Vector2;
    
    getSpawnPosition(): Vector2 {
        if (this.rallyPoint) {
            // Spawn towards rally point
            const direction = normalize(
                subtract(this.rallyPoint, this.position)
            );
            const distance = magnitude(this.defaultSpawnOffset);
            return add(this.position, scale(direction, distance));
        } else {
            // DEFAULT: Fixed offset (FIXES BUG)
            return add(this.position, this.defaultSpawnOffset);
        }
    }
    
    spawnUnit(): Unit {
        const unit = new Unit();
        unit.position = this.getSpawnPosition(); // ✅ Outside building
        
        if (this.rallyPoint) {
            unit.moveTo(this.rallyPoint);
        }
        
        return unit;
    }
}
```

## Server-Side Implementation

If spawning happens server-side (like in `GameServer.js`):

```javascript
// GameServer.js
class GameServer {
    spawnUnit(buildingId, playerId) {
        const building = this.getBuilding(buildingId);
        
        // Calculate spawn position (OUTSIDE building)
        const spawnPos = this.getSpawnPosition(building);
        
        const unit = {
            id: generateId(),
            type: building.unitType,
            position: spawnPos,  // ✅ Not inside building
            owner: playerId,
            rallyPoint: building.rallyPoint || null
        };
        
        this.units.push(unit);
        this.broadcastUnitSpawned(unit);
        
        return unit;
    }
    
    getSpawnPosition(building) {
        if (building.rallyPoint) {
            // Spawn towards rally point
            const direction = normalize(
                subtract(building.rallyPoint, building.position)
            );
            const distance = magnitude(building.defaultSpawnOffset);
            return add(building.position, scale(direction, distance));
        } else {
            // DEFAULT: Fixed offset (FIXES THE BUG)
            return add(building.position, building.defaultSpawnOffset);
        }
    }
}
```

## Configuration Examples

### Different Building Types

```gdscript
# Barracks - units exit from the right
barracks.default_spawn_offset = Vector2(80, 0)

# Factory - units exit from the bottom
factory.default_spawn_offset = Vector2(0, 100)

# Airfield - units spawn above and away
airfield.default_spawn_offset = Vector2(0, -150)

# Naval Shipyard - ships spawn in front in water
shipyard.default_spawn_offset = Vector2(200, 0)
```

### Direction-Based Spawning

```gdscript
# Spawn based on building rotation
func get_spawn_position() -> Vector2:
    var forward = Vector2.RIGHT.rotated(rotation)
    var offset = forward * default_spawn_offset.length()
    return global_position + offset
```

## Testing Checklist

- [ ] Units spawn outside building bounds
- [ ] Units are immediately visible after spawn
- [ ] Units are selectable after spawn
- [ ] Works without rally point set
- [ ] Works with rally point set
- [ ] Units move to rally point (if set)
- [ ] Multiple buildings don't overlap spawn points
- [ ] Spawn position is walkable/valid terrain

## Migration Guide

### For Existing Codebases

1. **Identify spawn code:**
   ```bash
   grep -r "spawn.*position.*building" .
   grep -r "unit.*position.*=.*building" .
   ```

2. **Add default offset property:**
   ```gdscript
   # Add to each building class
   @export var default_spawn_offset := Vector2(100, 0)
   ```

3. **Update spawn calls:**
   ```gdscript
   # OLD:
   unit.position = building.position
   
   # NEW:
   unit.position = building.get_spawn_position()
   ```

4. **Test each building type** to ensure offset is appropriate

### Backward Compatibility

If you need to support old saves/replays:

```gdscript
func get_spawn_position() -> Vector2:
    # Use legacy behavior if offset not set
    if default_spawn_offset.length() < 1.0:
        return global_position  # Old behavior
    
    # New behavior (fixed)
    if has_rally_point:
        var direction = (rally_point - global_position).normalized()
        return global_position + direction * default_spawn_offset.length()
    else:
        return global_position + default_spawn_offset
```

## Performance Considerations

This fix has negligible performance impact:
- One vector addition per spawn
- Optional direction calculation if rally point exists
- No additional memory overhead

## Common Pitfalls

### ❌ Don't hardcode spawn position

```gdscript
# BAD:
unit.position = Vector2(500, 300)  # What if building moves?
```

### ✅ Always relative to building

```gdscript
# GOOD:
unit.position = building.position + offset
```

### ❌ Don't forget collision

```gdscript
# BAD: Spawn might still be in collision
spawn_pos = building.position + offset
```

### ✅ Check spawn validity

```gdscript
# GOOD:
spawn_pos = building.position + offset
spawn_pos = find_nearest_valid_position(spawn_pos)
```

## Additional Improvements

Consider these enhancements:

1. **Multiple spawn points:**
   ```gdscript
   var spawn_points = [Vector2(80, 0), Vector2(0, 80), Vector2(-80, 0)]
   var spawn_pos = building.position + spawn_points[spawn_index]
   ```

2. **Dynamic formation:**
   ```gdscript
   func get_spawn_position_for_unit(index: int) -> Vector2:
       var formation_offset = calculate_formation_offset(index)
       return building.position + default_spawn_offset + formation_offset
   ```

3. **Terrain validation:**
   ```gdscript
   var spawn_pos = get_spawn_position()
   if not is_position_valid(spawn_pos):
       spawn_pos = find_nearest_valid_position(spawn_pos)
   ```

## Conclusion

This fix ensures units always spawn at a visible, accessible location outside buildings, significantly improving the player experience. The solution is:

- ✅ Simple to implement
- ✅ Performance-friendly
- ✅ Works with existing rally point systems
- ✅ Easy to configure per building type
- ✅ Backward compatible (with care)

For questions or issues, refer to the example implementation in `/examples/rts_unit_spawning/`.
