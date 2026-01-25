# RTSCamera Usage Examples

## Quick Start

### Example 1: Basic Integration

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function MyGame() {
  return (
    <RTSCamera>
      <div style={{ width: '2000px', height: '2000px' }}>
        {/* Your game content here */}
      </div>
    </RTSCamera>
  )
}
```

### Example 2: Custom Speed Settings

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function FastGame() {
  return (
    <RTSCamera
      basePanSpeed={10}
      shiftSpeedMultiplier={3}
    >
      <GameWorld />
    </RTSCamera>
  )
}
```

### Example 3: Position Tracking

```tsx
import { useState } from 'react'
import RTSCamera from '@/src/camera/RTSCamera'

function TrackedGame() {
  const [position, setPosition] = useState({ x: 0, y: 0 })

  const handlePositionChange = (x: number, y: number) => {
    setPosition({ x, y })
    // Load new chunks, update minimap, etc.
  }

  return (
    <>
      <div className="minimap">
        Camera at: {position.x}, {position.y}
      </div>
      <RTSCamera onPositionChange={handlePositionChange}>
        <GameWorld />
      </RTSCamera>
    </>
  )
}
```

### Example 4: Integrating with Orca Engine

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function OrcaRTSGame() {
  const [engineReady, setEngineReady] = useState(false)

  return (
    <div className="game-container">
      <RTSCamera
        basePanSpeed={7}
        shiftSpeedMultiplier={2.5}
        onPositionChange={(x, y) => {
          // Update Orca engine camera position
          if (engineReady) {
            window.orcaEngine?.setCameraPosition(x, y)
          }
        }}
      >
        <iframe
          src="/orca-engine"
          className="game-viewport"
          onLoad={() => setEngineReady(true)}
        />
      </RTSCamera>
    </div>
  )
}
```

### Example 5: Minimap Integration

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function GameWithMinimap() {
  const [cameraPos, setCameraPos] = useState({ x: 0, y: 0 })
  const WORLD_WIDTH = 4000
  const WORLD_HEIGHT = 4000
  const VIEWPORT_WIDTH = 800
  const VIEWPORT_HEIGHT = 600

  return (
    <div className="relative">
      {/* Minimap */}
      <div className="absolute top-4 right-4 w-48 h-48 bg-black/80 border-2 border-white z-50">
        <div className="relative w-full h-full">
          {/* World miniature */}
          <div className="w-full h-full bg-gray-800" />
          
          {/* Camera viewport indicator */}
          <div
            className="absolute border-2 border-yellow-400"
            style={{
              left: `${(cameraPos.x / WORLD_WIDTH) * 100}%`,
              top: `${(cameraPos.y / WORLD_HEIGHT) * 100}%`,
              width: `${(VIEWPORT_WIDTH / WORLD_WIDTH) * 100}%`,
              height: `${(VIEWPORT_HEIGHT / WORLD_HEIGHT) * 100}%`
            }}
          />
        </div>
      </div>

      {/* Main game view */}
      <RTSCamera onPositionChange={(x, y) => setCameraPos({ x, y })}>
        <GameWorld width={WORLD_WIDTH} height={WORLD_HEIGHT} />
      </RTSCamera>
    </div>
  )
}
```

## Testing the Camera

To test the camera implementation:

1. Navigate to `/camera-demo` in your browser
2. Use WASD or arrow keys to move around
3. Hold SHIFT while moving to see the speed boost
4. Adjust sliders to change speed settings in real-time
5. Check the debug overlay for position and speed info

## Performance Tips

### Optimizing for Large Worlds

```tsx
import { useMemo } from 'react'
import RTSCamera from '@/src/camera/RTSCamera'

function LargeWorldGame() {
  const visibleChunks = useMemo(() => {
    // Calculate which chunks are visible based on camera position
    return calculateVisibleChunks(cameraPos)
  }, [cameraPos])

  return (
    <RTSCamera onPositionChange={updateCameraPos}>
      {visibleChunks.map(chunk => (
        <Chunk key={chunk.id} {...chunk} />
      ))}
    </RTSCamera>
  )
}
```

### Debouncing Position Updates

```tsx
import { useCallback, useRef } from 'react'
import RTSCamera from '@/src/camera/RTSCamera'

function OptimizedGame() {
  const timeoutRef = useRef<NodeJS.Timeout>()

  const handlePositionChange = useCallback((x: number, y: number) => {
    // Debounce expensive operations
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(() => {
      updateGameState(x, y)
    }, 100)
  }, [])

  return (
    <RTSCamera onPositionChange={handlePositionChange}>
      <GameWorld />
    </RTSCamera>
  )
}
```

## Common Patterns

### Constraining Camera Bounds

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function BoundedGame() {
  const MIN_X = 0
  const MAX_X = 2000
  const MIN_Y = 0
  const MAX_Y = 2000

  const handlePositionChange = (x: number, y: number) => {
    const boundedX = Math.max(MIN_X, Math.min(MAX_X, x))
    const boundedY = Math.max(MIN_Y, Math.min(MAX_Y, y))
    
    // Apply bounds or provide feedback
    if (x !== boundedX || y !== boundedY) {
      console.log('Camera reached bounds')
    }
  }

  return (
    <RTSCamera onPositionChange={handlePositionChange}>
      <GameWorld />
    </RTSCamera>
  )
}
```

## Keyboard Control Specifications

| Input | Action | Speed Calculation |
|-------|--------|-------------------|
| W or ↑ | Move Up | `y -= basePanSpeed` |
| S or ↓ | Move Down | `y += basePanSpeed` |
| A or ← | Move Left | `x -= basePanSpeed` |
| D or → | Move Right | `x += basePanSpeed` |
| SHIFT + Any | Fast Move | `speed *= shiftSpeedMultiplier` |

### Diagonal Movement

When moving diagonally (e.g., W+D), the speed is normalized:

```
normalizedSpeed = speed / √2
```

This ensures that diagonal movement speed equals the speed of cardinal directions.

## Feature Checklist

- [x] WASD key support
- [x] Arrow key support  
- [x] SHIFT speed boost (2-3x)
- [x] Smooth movement
- [x] Normalized diagonal movement
- [x] Position tracking callback
- [x] Debug mode in development
- [x] Customizable speeds
- [x] TypeScript support
- [x] React 18+ compatibility

## Browser Support

The component works in all modern browsers that support:
- ES6+ JavaScript
- React 18+
- `requestAnimationFrame`
- Keyboard events (KeyboardEvent API)

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
