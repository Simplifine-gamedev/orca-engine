# Integrating RTS Camera with Orca IDE

This guide shows how to integrate the edge-panning camera controls into the Orca IDE viewport.

## Quick Integration

### Option 1: Wrap the Viewport (Recommended)

Wrap your 3D viewport iframe with the `RTSCamera` component:

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

// In your IDE component
<RTSCamera
  config={{
    edgeThreshold: 40,
    baseSpeed: 8,
    maxSpeed: 25,
  }}
  onCameraMove={(position) => {
    // Send camera position to your game engine
    if (socket) {
      socket.emit('camera-update', position)
    }
  }}
  className="w-1/2 bg-black relative"
>
  {/* Your existing VNC iframe */}
  {vncUrl && (
    <iframe
      ref={iframeRef}
      src={`${vncUrl}?autoconnect=true&resize=scale&quality=6`}
      className="w-full h-full"
      style={{ border: 'none' }}
    />
  )}
</RTSCamera>
```

### Option 2: Use the Hook for Custom Control

For more control over the integration:

```tsx
import { useRef } from 'react'
import { useRTSCamera } from '@/src/camera/RTSCamera'

function Viewport() {
  const viewportRef = useRef<HTMLDivElement>(null)
  const { position, velocity, enabled, setEnabled } = useRTSCamera(
    viewportRef,
    {
      edgeThreshold: 40,
      baseSpeed: 8,
      maxSpeed: 25,
    }
  )

  // Send position updates to game engine
  useEffect(() => {
    if (socket && (velocity.x !== 0 || velocity.y !== 0)) {
      socket.emit('camera-update', position)
    }
  }, [position, velocity, socket])

  return (
    <div ref={viewportRef} className="relative w-full h-full">
      {/* Your viewport content */}
      <iframe ... />
      
      {/* Optional: Toggle button */}
      <button
        onClick={() => setEnabled(!enabled)}
        className="absolute top-2 right-2 ..."
      >
        Camera Pan: {enabled ? 'ON' : 'OFF'}
      </button>
    </div>
  )
}
```

## Backend Integration

On the backend/game engine side, handle the camera position updates:

### WebSocket Handler (Python)

```python
@socketio.on('camera-update')
def handle_camera_update(data):
    position = data['position']
    # Update the game engine camera
    # This depends on your engine implementation
    update_camera_position(position['x'], position['y'], position['z'])
```

### GDScript (Godot)

```gdscript
extends Camera3D

var target_position = Vector3.ZERO
var smooth_speed = 5.0

func _ready():
    # Connect to WebSocket or other communication channel
    pass

func update_camera_from_web(x: float, y: float, z: float):
    target_position = Vector3(x, y, z)

func _process(delta):
    # Smoothly move camera to target position
    global_position = global_position.lerp(target_position, smooth_speed * delta)
```

## Configuration Tips

### For RTS Games

```tsx
config={{
  edgeThreshold: 50,  // Medium threshold
  baseSpeed: 5,       // Moderate speed
  maxSpeed: 20,       // Fast at edges
  enabled: true,
}}
```

### For Strategy Games with Slow Pacing

```tsx
config={{
  edgeThreshold: 60,  // Larger threshold
  baseSpeed: 3,       // Slower speed
  maxSpeed: 12,       // Moderate max speed
  enabled: true,
}}
```

### For Fast-Paced Action

```tsx
config={{
  edgeThreshold: 40,  // Smaller threshold
  baseSpeed: 10,      // Fast speed
  maxSpeed: 30,       // Very fast at edges
  enabled: true,
}}
```

## User Settings

Add a settings panel to let users customize the camera:

```tsx
function CameraSettings() {
  const [config, setConfig] = useState({
    edgeThreshold: 50,
    baseSpeed: 5,
    maxSpeed: 20,
    enabled: true,
  })

  return (
    <div className="settings-panel">
      <h3>Camera Settings</h3>
      
      <label>
        Enable Edge Panning
        <input
          type="checkbox"
          checked={config.enabled}
          onChange={(e) => setConfig({
            ...config,
            enabled: e.target.checked
          })}
        />
      </label>

      <label>
        Edge Sensitivity (px)
        <input
          type="range"
          min="20"
          max="100"
          value={config.edgeThreshold}
          onChange={(e) => setConfig({
            ...config,
            edgeThreshold: parseInt(e.target.value)
          })}
        />
      </label>

      <label>
        Camera Speed
        <input
          type="range"
          min="1"
          max="30"
          value={config.maxSpeed}
          onChange={(e) => setConfig({
            ...config,
            maxSpeed: parseInt(e.target.value)
          })}
        />
      </label>
    </div>
  )
}
```

## Testing

To test the integration:

1. Run the development server: `npm run dev`
2. Navigate to `/camera-demo` to see examples
3. Move your mouse to the edges of the viewport
4. Verify the camera position updates are being logged
5. Check that the toggle button works

## Troubleshooting

### Camera not responding

- Check that the `containerRef` is attached to the correct element
- Verify `enabled` is set to `true`
- Make sure the container has defined dimensions

### Jerky movement

- Adjust the `smoothing` parameter (default: 0.85)
- Check frame rate - the component uses `requestAnimationFrame`
- Verify no competing event handlers are interfering

### Incorrect edge detection

- Check container positioning (should be `relative` or `absolute`)
- Verify no CSS transforms are affecting the container
- Ensure the container fills its intended space

## Performance

The RTS camera is optimized for performance:

- Uses `requestAnimationFrame` for smooth 60fps movement
- Delta time normalization for consistent speed across frame rates
- Minimal DOM queries (cached container dimensions during mouse move)
- No re-renders unless position/velocity actually changes

## Next Steps

- Save user camera preferences to localStorage
- Add keyboard shortcuts for toggling edge panning
- Implement camera zoom with mouse wheel
- Add minimap click-to-move integration
