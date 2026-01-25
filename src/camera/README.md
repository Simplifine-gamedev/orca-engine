# RTS Camera Component

An RTS-style camera component with zoom, pan, and rotation controls for React applications using Three.js.

## Features

- **Zoom**: Mouse wheel to zoom in/out with configurable min/max distances
- **Pan**: Left-click and drag to pan the camera
- **Rotate**: Right-click/Ctrl+drag to rotate the camera
- **Configurable Constraints**: Customize zoom limits and initial positioning

## Usage

```tsx
import { useRef } from 'react';
import { RTSCamera } from './camera/RTSCamera';

function GameScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  return (
    <>
      <canvas ref={canvasRef} />
      <RTSCamera 
        canvasRef={canvasRef}
        minDistance={1}    // Allow close zoom (default: 1)
        maxDistance={50}   // Maximum zoom out (default: 50)
        initialDistance={20} // Starting distance (default: 20)
      />
    </>
  );
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `canvasRef` | `React.RefObject<HTMLCanvasElement>` | Required | Reference to the canvas element |
| `minDistance` | `number` | `1` | Minimum zoom distance (closer = lower value) |
| `maxDistance` | `number` | `50` | Maximum zoom distance |
| `initialDistance` | `number` | `20` | Initial camera distance |

## Controls

- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan camera
- **Right Click + Drag** or **Ctrl + Drag**: Rotate camera view
- **Pitch**: Automatically constrained between 0.1 and ~89 degrees

## Recent Changes

### ORC-207: Increased Max Zoom
- Reduced default `minDistance` from 5 to 1
- Allows users to zoom in 5x closer to the action
- Addresses user feedback requesting closer camera zoom

## Dependencies

- React
- Three.js
