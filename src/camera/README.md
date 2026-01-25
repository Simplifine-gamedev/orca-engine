# RTS Camera Component

RTS-style camera controller for Orca RTS games with zoom and pan controls.

## Features

- **Zoom Control**: Mouse wheel to zoom in/out
- **Pan Control**: WASD keys, arrow keys, or middle-mouse drag
- **Configurable Limits**: Adjustable min/max zoom distances
- **RTS Perspective**: 45-degree angled top-down view

## Usage

```typescript
import { RTSCamera } from './camera/RTSCamera';

function GameComponent() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  return (
    <>
      <canvas ref={canvasRef} />
      <RTSCamera 
        canvasRef={canvasRef}
        minDistance={2}    // Minimum zoom distance (closer = more zoomed in)
        maxDistance={50}   // Maximum zoom distance
        zoomSpeed={2}      // Zoom sensitivity
        moveSpeed={0.5}    // Pan speed
      />
    </>
  );
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `canvasRef` | `React.RefObject<HTMLCanvasElement>` | required | Reference to the canvas element |
| `minDistance` | `number` | `2` | Minimum camera distance (allows closer zoom) |
| `maxDistance` | `number` | `50` | Maximum camera distance |
| `zoomSpeed` | `number` | `2` | Mouse wheel zoom sensitivity |
| `moveSpeed` | `number` | `0.5` | Keyboard/mouse pan speed |

## Controls

- **Mouse Wheel**: Zoom in/out
- **W/↑**: Move forward
- **S/↓**: Move backward
- **A/←**: Move left
- **D/→**: Move right
- **Middle Mouse / Shift+Left Mouse**: Drag to pan

## Recent Changes

### v1.0.0 - Increased Zoom Range
- **Reduced `minDistance` to 2** (from 5) to allow players to zoom in much closer to the action
- Addresses user feedback: "zoom in more" (Gaudio)
- Allows better detail viewing of units and buildings

## Dependencies

- React
- Three.js

## Notes

- Camera maintains a 45-degree angle typical of RTS games
- All movements are smooth and interpolated
- Zoom limits prevent camera from clipping through terrain or zooming too far out
