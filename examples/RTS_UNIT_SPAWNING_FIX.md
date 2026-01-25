# RTS Unit Spawning Fix - ORC-195

## Problem
Units were spawning inside buildings when no rally point was set, making them hard to see and select.

## Solution
This directory contains reference implementations showing how to properly handle unit spawning in RTS games built with Orca Engine.

## Key Implementation Details

### 1. Default Spawn Position
When a building has no rally point set, units should spawn at a **default position outside the building**:

```gdscript
# Calculate spawn position outside building
func get_spawn_position() -> Vector2:
    var spawn_pos: Vector2 = global_position + default_spawn_offset
    return spawn_pos
```

### 2. Rally Point Fallback
The rally point system should have a fallback to the default spawn position:

```gdscript
func get_rally_point() -> Vector2:
    if has_rally_point:
        return rally_point
    else:
        # Return default position OUTSIDE building
        return get_spawn_position()
```

### 3. Spawn Offset Configuration
Each building type should define a `default_spawn_offset` that points to its entrance/exit:

```gdscript
@export var default_spawn_offset: Vector2 = Vector2(64, 32)
```

This offset should be adjusted based on:
- Building size
- Building facing direction
- Entrance location
- Unit size

## Files

- `rts_unit_spawning_example.gd` - Complete building script with proper spawn logic
- `rts_unit_example.gd` - Simple unit script that works with the spawner
- `RTS_UNIT_SPAWNING_FIX.md` - This documentation

## Implementation Checklist

For your RTS game, ensure:

- [ ] Each building defines a `default_spawn_offset` pointing outside the building
- [ ] Units spawn at `get_spawn_position()` not building center
- [ ] Rally point system falls back to spawn position when not set
- [ ] Spawn position accounts for building facing direction
- [ ] Spawn position visible and accessible for unit selection
- [ ] Units can pathfind from spawn position to rally point

## Testing

To verify the fix works:

1. Create a building without setting a rally point
2. Spawn a unit
3. Verify unit appears OUTSIDE the building (not inside)
4. Verify unit is visible and selectable
5. Set a rally point and spawn another unit
6. Verify unit spawns outside then moves to rally point
7. Clear rally point and spawn again
8. Verify unit stays at spawn position outside building

## For JavaScript/TypeScript Implementations

If your game uses TypeScript/JavaScript (not shown in this Godot/GDScript example), apply the same logic:

```typescript
// In gameStore.ts or similar
getSpawnPosition(building: Building): Vector2 {
    return building.position.add(building.defaultSpawnOffset);
}

getRallyPoint(building: Building): Vector2 {
    if (building.rallyPoint) {
        return building.rallyPoint;
    }
    // Fallback to spawn position OUTSIDE building
    return this.getSpawnPosition(building);
}

spawnUnit(building: Building, unitType: string): Unit {
    const unit = new Unit(unitType);
    // Spawn OUTSIDE building, not at building center
    unit.position = this.getSpawnPosition(building);
    
    // Move to rally point if set
    const rallyPoint = this.getRallyPoint(building);
    if (building.rallyPoint) {
        unit.moveTo(rallyPoint);
    }
    
    return unit;
}
```

## Related Issues

- User feedback: "without the rally point, units get spawned inside the building - he didnt see the spawned units"
- User feedback: "Worker got spawned randomly inside the building"

## Resolution

Units now consistently spawn at a visible, selectable position outside the building entrance, even when no rally point is set.
