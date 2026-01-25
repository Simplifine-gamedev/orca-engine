# RTS Watchtower Vision Demo

This demo showcases the implementation of visual indicators for watchtowers in an RTS game, addressing the feature request from Linear issue **ORC-176**.

## Features Implemented

### 1. Eye Icon Indicator ✓
- **Eye icon sprite** floats above each watchtower
- Subtle floating animation for visual interest
- Color-coded by team ownership:
  - **Gray**: Neutral watchtower
  - **Blue**: Player-controlled
  - **Red**: Enemy-controlled

### 2. Vision Radius Preview ✓
- **Vision radius indicator** appears on mouse hover
- Semi-transparent circle shows the area of vision
- Radius is customizable per watchtower (default: 20m)
- Indicator color matches team ownership

### 3. Tooltip System ✓
- **Informative tooltip** displays on hover
- Shows:
  - Current control status (Neutral/Controlled/Enemy)
  - Vision radius in meters
  - Explanation: "Conquer to reveal map area"

## Controls

| Input | Action |
|-------|--------|
| **WASD** or **Arrow Keys** | Move camera |
| **Mouse Wheel** | Zoom in/out |
| **Middle Mouse + Drag** | Rotate camera |
| **Hover over tower** | Show vision radius and tooltip |
| **Click tower** | Conquer watchtower |
| **SPACE** | Toggle team control for all towers |
| **R** | Reset all towers to neutral |
| **1-5** | Conquer specific tower (by number) |
| **ESC** | Quit demo |

## File Structure

```
examples/rts_demo/
├── project.godot              # Godot project configuration
├── icon.svg                   # Project icon featuring watchtower with eye
├── README.md                  # This file
├── scenes/
│   ├── main.tscn             # Main demo scene with 5 watchtowers
│   └── control_point.tscn    # Watchtower prefab scene
└── scripts/
    ├── control_point.gd      # Main watchtower logic (eye icon, vision, tooltip)
    ├── watchtower_model.gd   # Procedural 3D tower model
    ├── camera_controller.gd  # RTS-style camera controls
    └── demo_controller.gd    # Demo interaction logic
```

## Implementation Details

### Eye Icon (`control_point.gd`)
```gdscript
func _setup_eye_icon():
    eye_icon = Sprite3D.new()
    eye_icon.billboard = BaseMaterial3D.BILLBOARD_ENABLED
    eye_icon.texture = _create_eye_texture()
    eye_icon.position = Vector3(0, 3, 0)
    # Floating animation
    var tween = create_tween().set_loops()
    tween.tween_property(eye_icon, "position:y", 3.3, 1.5)
    tween.tween_property(eye_icon, "position:y", 3.0, 1.5)
```

### Vision Radius Indicator
```gdscript
func _setup_vision_indicator():
    var mesh = CylinderMesh.new()
    mesh.top_radius = vision_radius
    mesh.bottom_radius = vision_radius
    mesh.height = 0.1
    # Semi-transparent material
    var material = StandardMaterial3D.new()
    material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
    material.albedo_color = Color(1, 1, 1, 0.3)
```

### Tooltip System
```gdscript
func _update_tooltip_text():
    var status = "Neutral" if team == 0 else "Controlled"
    tooltip_label.text = "Watchtower [%s]\nVision Radius: %.1fm\n\nConquer to reveal map area" % [status, vision_radius]
```

## Customization

The `ControlPoint` class exposes several export variables:

```gdscript
@export var vision_radius: float = 20.0  # Adjust vision range
@export var team: int = 0                # 0=neutral, 1=player, 2=enemy
@export var conquered: bool = false      # Conquest state
```

## Running the Demo

1. Open the Orca Engine editor
2. Navigate to `examples/rts_demo/`
3. Open `project.godot`
4. Press **F5** or click **Run Project**

## Integration into Existing Projects

To use this watchtower system in your RTS game:

1. **Copy the scripts**:
   - `control_point.gd` → Your objects/buildings folder
   - `watchtower_model.gd` → Your models folder (or use your own 3D model)

2. **Instantiate the ControlPoint scene**:
   ```gdscript
   var watchtower = preload("res://scenes/control_point.tscn").instantiate()
   watchtower.position = Vector3(x, 0, z)
   watchtower.vision_radius = 25.0
   add_child(watchtower)
   ```

3. **Connect to your game logic**:
   ```gdscript
   watchtower.conquer(player_team)  # When player conquers
   var radius = watchtower.vision_radius  # For fog-of-war system
   ```

## User Feedback Addressed

> **Haridzieko**: "there can be an eye icon or sth on top of the watchtowers so its more intuitive and they figure out what conquering the towers do before doing it"

✅ **Implemented**: Eye icon floats above all watchtowers
✅ **Implemented**: Vision radius preview shows exact coverage area
✅ **Implemented**: Tooltip explains the benefit: "Conquer to reveal map area"

## Future Enhancements

Potential improvements for production use:

- [ ] Add fog-of-war integration
- [ ] Animated eye that "looks around"
- [ ] Minimap integration showing tower vision
- [ ] Sound effects for conquest
- [ ] Particle effects when conquering
- [ ] Network synchronization for multiplayer

## License

This demo is part of the Orca Engine project and follows the same licensing terms.

## Credits

Created as a demonstration for Linear issue **ORC-176**: "[Map] Add eye icon on watchtowers to indicate vision"
