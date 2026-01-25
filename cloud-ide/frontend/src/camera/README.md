# RTS Camera

Edge-of-screen camera panning for RTS-style games.

## Features

- ✅ Detects mouse position near screen edges
- ✅ Automatic camera panning when mouse approaches edges
- ✅ Variable speed based on distance to edge
- ✅ Works on all 4 edges (top, bottom, left, right)
- ✅ Enable/disable toggle
- ✅ Configurable threshold, speed, and smoothing
- ✅ Frame-independent movement

## Usage

### As a Hook

```tsx
import { useRTSCamera } from './camera/RTSCamera'

function GameViewport() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { position, velocity, enabled, setEnabled } = useRTSCamera(
    containerRef,
    {
      edgeThreshold: 50,  // Trigger panning within 50px of edge
      baseSpeed: 5,       // Base panning speed
      maxSpeed: 20,       // Max speed at edge
      enabled: true,      // Start with panning enabled
    }
  )

  return (
    <div ref={containerRef} className="w-full h-full">
      <Canvas camera={{ position: [position.x, position.y, position.z] }}>
        {/* Your 3D scene */}
      </Canvas>
    </div>
  )
}
```

### As a Component

```tsx
import RTSCamera from './camera/RTSCamera'

function Game() {
  const handleCameraMove = (position) => {
    console.log('Camera moved to:', position)
    // Update your game camera here
  }

  return (
    <RTSCamera
      config={{
        edgeThreshold: 50,
        baseSpeed: 5,
        maxSpeed: 20,
        enabled: true,
      }}
      onCameraMove={handleCameraMove}
      className="game-viewport"
    >
      {/* Your game content */}
      <div className="w-full h-full bg-gray-900">
        Game viewport content
      </div>
    </RTSCamera>
  )
}
```

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `edgeThreshold` | number | 50 | Distance from edge (in pixels) to trigger panning |
| `baseSpeed` | number | 5 | Minimum panning speed when at threshold |
| `maxSpeed` | number | 20 | Maximum panning speed at screen edge |
| `enabled` | boolean | true | Enable/disable edge panning |
| `smoothing` | number | 0.85 | Smoothing factor for camera movement (0-1) |

## Behavior

The camera panning speed increases linearly as the mouse gets closer to the screen edge:

- At `edgeThreshold` distance: pans at `baseSpeed`
- At screen edge (0 pixels): pans at `maxSpeed`
- Between threshold and edge: interpolates between base and max speed

## Controls

- **Move mouse to screen edges**: Camera pans in that direction
- **Click toggle button (top-right)**: Enable/disable edge panning
- **Move mouse away from edges**: Camera stops panning

## Development

The component includes a debug overlay in development mode showing:
- Current camera position
- Current velocity
- Edge panning status

To disable the debug overlay in production, ensure `NODE_ENV` is set to `production`.
