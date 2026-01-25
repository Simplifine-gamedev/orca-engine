# RTS Camera Edge Panning Feature

## Overview

This feature implements edge-of-screen camera panning for RTS-style games. When the mouse moves near the edges of the viewport, the camera automatically pans in that direction.

## What Was Implemented

### 1. RTSCamera Component (`app/components/RTSCamera.tsx`)
A reusable React component that provides:
- Edge detection on all 4 sides (top, bottom, left, right)
- Variable pan speed based on distance from edge
- Configurable thresholds and speeds
- Smooth 60fps animation using requestAnimationFrame
- Enable/disable toggle
- Visual debug indicators in development mode

### 2. Interactive Demo (`app/demo/camera/page.tsx`)
A full-featured demo page with:
- Real-time camera panning visualization
- Interactive settings panel
- Sample game objects that move with camera
- Grid background for visual feedback
- Position tracking display

### 3. Documentation (`app/components/README.md`)
Complete documentation including:
- Usage examples
- API reference
- Props documentation
- Performance notes

## Accessing the Feature

### Demo Page
Visit `/demo/camera` to see an interactive demonstration:
```
http://localhost:3000/demo/camera
```

### Using in Your Code
```tsx
import RTSCamera from '@/app/components/RTSCamera'

function MyGame() {
  const [cameraPos, setCameraPos] = useState({ x: 0, y: 0 })

  return (
    <RTSCamera
      enabled={true}
      edgeThreshold={50}
      maxPanSpeed={10}
      onCameraMove={(dx, dy) => {
        setCameraPos(prev => ({ x: prev.x + dx, y: prev.y + dy }))
      }}
    >
      {/* Your game content */}
    </RTSCamera>
  )
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Toggle edge panning on/off |
| `edgeThreshold` | `50` | Distance from edge (px) to trigger panning |
| `maxPanSpeed` | `10` | Maximum pan speed (px/frame) |
| `minPanSpeed` | `2` | Minimum pan speed (px/frame) |

## Technical Details

- **Performance**: Uses requestAnimationFrame for smooth animation
- **Cleanup**: Automatically cancels animation frames on unmount
- **Responsive**: Adapts to window resize events
- **Type-safe**: Full TypeScript support
- **Zero dependencies**: Only uses React built-in hooks

## Testing

The implementation has been:
- ✓ Built successfully with Next.js production build
- ✓ Linted with no errors
- ✓ Type-checked with TypeScript
- ✓ Tested with the interactive demo

## Future Enhancements

Potential improvements for future iterations:
- Add diagonal movement optimization
- Implement momentum/easing effects
- Add keyboard shortcuts for camera control
- Support touch/mobile devices
- Add zoom controls
- Implement camera bounds/limits
- Add smooth camera centering on objects
