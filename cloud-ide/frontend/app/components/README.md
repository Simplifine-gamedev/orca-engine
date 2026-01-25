# RTSCamera Component

Edge-of-screen camera panning component for RTS-style games.

## Features

- ✓ Detects mouse position near screen edges
- ✓ Pans camera in the appropriate direction
- ✓ Speed increases closer to the edge
- ✓ Works on all 4 edges (top, bottom, left, right)
- ✓ Configurable thresholds and speeds
- ✓ Easy to enable/disable
- ✓ Smooth animation using requestAnimationFrame
- ✓ Visual debug indicators in development mode

## Usage

```tsx
import RTSCamera from './components/RTSCamera'

function MyGame() {
  const [cameraPos, setCameraPos] = useState({ x: 0, y: 0 })

  const handleCameraMove = (deltaX: number, deltaY: number) => {
    setCameraPos(prev => ({
      x: prev.x + deltaX,
      y: prev.y + deltaY
    }))
  }

  return (
    <RTSCamera
      enabled={true}
      edgeThreshold={50}
      maxPanSpeed={10}
      minPanSpeed={2}
      onCameraMove={handleCameraMove}
      className="w-full h-full"
    >
      {/* Your game viewport content here */}
      <div style={{ transform: `translate(${cameraPos.x}px, ${cameraPos.y}px)` }}>
        {/* Game objects */}
      </div>
    </RTSCamera>
  )
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `boolean` | `true` | Enable or disable edge panning |
| `edgeThreshold` | `number` | `50` | Distance from edge in pixels to trigger panning |
| `maxPanSpeed` | `number` | `10` | Maximum pan speed in pixels per frame |
| `minPanSpeed` | `number` | `2` | Minimum pan speed in pixels per frame |
| `onCameraMove` | `(deltaX: number, deltaY: number) => void` | - | Callback when camera position changes |
| `className` | `string` | `''` | CSS class for the container |
| `children` | `React.ReactNode` | - | Content to render inside the viewport |

## How It Works

1. The component tracks mouse position relative to its container
2. When the mouse is within `edgeThreshold` pixels of any edge, it calculates a pan speed
3. The pan speed increases linearly from `minPanSpeed` to `maxPanSpeed` as the mouse gets closer to the edge
4. The `onCameraMove` callback is called every frame with delta values
5. Your application updates the camera position based on these deltas

## Demo

See `/demo/camera` for a full interactive demo with configurable settings.

## Performance

- Uses `requestAnimationFrame` for smooth 60fps animation
- Only updates when mouse is within edge zones
- Automatically cleans up animation frames on unmount
- No unnecessary re-renders

## Debug Mode

In development mode, visual indicators show the edge zones where panning is triggered.
