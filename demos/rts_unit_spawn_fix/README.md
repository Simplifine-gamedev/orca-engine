# RTS Unit Spawn Fix - Preventing T-Pose Animation Issue

## Problem: ORC-119

Units were appearing in T-pose when spawning from buildings before transitioning to idle animation. This is a common issue in game development when animation systems aren't properly initialized before the first frame renders.

## Root Cause

The T-pose issue occurs when:

1. A unit is instantiated and added to the scene tree
2. The unit renders its first frame before the animation system is initialized
3. Without animation data, the skeletal mesh defaults to T-pose (the bind pose)
4. After 1-2 frames, the animation system initializes and transitions to idle

This creates a visible "flash" of T-pose that looks unprofessional and breaks immersion.

## Solution Overview

The fix involves ensuring the animation system is **fully initialized and set to idle state** before the first frame renders. This requires proper ordering of operations in both the unit and building scripts.

## Implementation

### 1. Unit Script (`rts_unit.gd`)

**Key Changes:**

```gdscript
func _ready() -> void:
    # CRITICAL: Initialize animation system FIRST
    _initialize_animation_system()
    
    # Set animation state BEFORE first physics frame
    _set_animation_state(UnitState.IDLE)
    
    # Ensure animation tree is active immediately
    if animation_tree:
        animation_tree.active = true
```

**Why this works:**
- `_ready()` is called before the first frame renders
- Animation components are found and initialized immediately
- The state machine is set to "idle" before rendering
- AnimationTree is activated right away, preventing bind pose display

### 2. Building Script (`rts_building.gd`)

**Key Changes:**

```gdscript
func spawn_unit() -> void:
    # STEP 1: Instantiate (but don't add to tree yet)
    var unit: RTSUnit = unit_scene.instantiate()
    
    # STEP 2: Set position BEFORE adding to tree
    unit.global_position = spawn_pos
    unit.rotation.y = rotation.y
    
    # STEP 3: Add to scene tree
    # This triggers unit._ready() which initializes animation
    get_tree().current_scene.add_child(unit)
    
    # STEP 4: Post-spawn setup (animation already initialized)
    unit.spawn_at_position(spawn_pos)
```

**Why this order matters:**
1. Setting position before adding to tree ensures the unit appears in the correct location immediately
2. Adding to tree triggers `_ready()` which initializes animations
3. By the time the first frame renders, animation state is already set to idle
4. No T-pose is ever visible

### 3. Animation State Machine Setup

The unit script includes robust animation system detection:

```gdscript
func _initialize_animation_system() -> void:
    # Find AnimationPlayer and AnimationTree
    animation_player = _find_animation_player(self)
    animation_tree = _find_animation_tree(self)
    
    if animation_tree:
        state_machine = animation_tree.get("parameters/playback")
        animation_tree.active = true
        
        if state_machine:
            # Force immediate idle transition
            state_machine.travel("idle")
    elif animation_player:
        # Fallback to direct animation playback
        if animation_player.has_animation("idle"):
            animation_player.play("idle")
```

## Common Mistakes (What NOT to Do)

### ❌ Wrong: Add to tree before setting position

```gdscript
var unit = unit_scene.instantiate()
get_tree().current_scene.add_child(unit)  # ← Added first
unit.global_position = spawn_pos  # ← Position set after
```

**Problem:** Unit may render one frame at origin (0,0,0) before moving

### ❌ Wrong: Don't initialize animation in _ready()

```gdscript
func _ready() -> void:
    # Other setup but no animation initialization
    current_health = max_health
    # ← Missing animation setup!
```

**Problem:** Unit will T-pose until animation is set elsewhere

### ❌ Wrong: Try to set animation before adding to tree

```gdscript
var unit = unit_scene.instantiate()
unit._set_animation_state(UnitState.IDLE)  # ← Called too early
get_tree().current_scene.add_child(unit)
```

**Problem:** `_ready()` hasn't been called yet, so animation system doesn't exist

## Testing the Fix

### Files Structure
```
demos/rts_unit_spawn_fix/
├── scripts/
│   ├── rts_unit.gd          # Unit with proper animation init
│   └── rts_building.gd      # Building with proper spawn order
├── scenes/
│   ├── rts_unit.tscn        # Unit scene
│   ├── rts_building.tscn    # Building scene
│   └── test_scene.tscn      # Test scene with auto-spawn
└── README.md
```

### How to Test

1. Open `test_scene.tscn` in Godot
2. Run the scene (F5)
3. Observe units spawning from the building
4. Units should appear in idle animation immediately
5. No T-pose should be visible at any point

### Verification Checklist

- [ ] Units spawn in idle animation
- [ ] No T-pose flash visible
- [ ] Units transition smoothly to movement animation
- [ ] Animation state persists through gameplay
- [ ] Works with both AnimationTree and AnimationPlayer

## Technical Details

### Animation System Initialization Order

1. `instantiate()` - Creates node instance in memory
2. `add_child()` - Adds to scene tree, triggers `_ready()`
3. `_ready()` executes:
   - Finds AnimationPlayer/AnimationTree
   - Sets state machine to idle
   - Activates animation tree
4. First frame renders - unit already in idle pose

### Why AnimationTree Must Be Active

The AnimationTree node has an `active` property that must be `true` for animations to play. If not activated in `_ready()`, the unit will display its bind pose (T-pose) until activated.

```gdscript
if animation_tree:
    animation_tree.active = true  # ← Critical!
```

### State Machine Travel vs Play

- `state_machine.travel("idle")` - Smoothly transitions between states
- `animation_player.play("idle")` - Directly plays animation (no blending)

For spawning, we use `travel()` because it handles blend trees properly.

## Best Practices

1. **Always initialize animations in `_ready()`**
2. **Set position before adding to tree**
3. **Activate AnimationTree immediately**
4. **Use state machine travel for smooth transitions**
5. **Provide fallbacks for different animation setups**

## Related Issues

This fix also prevents similar issues with:
- Units spawning from death (respawn)
- Units being teleported
- Units being created by scripting
- Any dynamic entity spawning

## Performance Notes

- Initializing animations in `_ready()` has negligible performance impact
- The animation system would initialize eventually anyway
- This just ensures it happens before rendering
- No additional memory or CPU overhead

## Compatibility

This fix works with:
- Godot 4.x (current version)
- AnimationTree with StateMachine
- Direct AnimationPlayer control
- Both 3D and 2D animated characters (with appropriate adjustments)

## References

- Linear Issue: ORC-119
- Files: `rts_unit.gd`, `rts_building.gd`
- Branch: `cursor/ORC-119-unit-t-pose-animation-9d8d`

## License

Same as parent project (Orca Engine).
