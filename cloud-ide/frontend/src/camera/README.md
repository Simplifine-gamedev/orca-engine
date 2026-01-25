# RTS Camera Component

A React component providing RTS-style camera controls with keyboard navigation and SHIFT key speed boost.

## Features

- **WASD Controls**: Move camera with W, A, S, D keys
- **Arrow Keys**: Alternative movement with arrow keys
- **SHIFT Speed Boost**: Hold SHIFT for 2-3x faster movement
- **Smooth Movement**: Normalized diagonal movement and smooth animations
- **Customizable**: Adjustable base speed and speed multiplier
- **Debug Mode**: Visual feedback in development mode

## Installation

The component is part of the Orca Cloud IDE frontend. No additional installation required.

## Usage

### Basic Usage

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function GameView() {
  return (
    <RTSCamera>
      <YourGameContent />
    </RTSCamera>
  )
}
```

### Advanced Usage

```tsx
import RTSCamera from '@/src/camera/RTSCamera'

function GameView() {
  const handlePositionChange = (x: number, y: number) => {
    console.log(`Camera moved to: ${x}, ${y}`)
    // Update game state, load new chunks, etc.
  }

  return (
    <RTSCamera
      basePanSpeed={7}
      shiftSpeedMultiplier={3}
      onPositionChange={handlePositionChange}
    >
      <YourGameContent />
    </RTSCamera>
  )
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `basePanSpeed` | `number` | `5` | Base camera movement speed in pixels per frame |
| `shiftSpeedMultiplier` | `number` | `2.5` | Speed multiplier when SHIFT is held (2-3x recommended) |
| `onPositionChange` | `(x: number, y: number) => void` | `undefined` | Callback fired when camera position changes |
| `children` | `React.ReactNode` | `undefined` | Content to render (typically the game viewport) |

## Keyboard Controls

| Key | Action |
|-----|--------|
| W / ↑ | Move camera up |
| S / ↓ | Move camera down |
| A / ← | Move camera left |
| D / → | Move camera right |
| SHIFT + Movement | Move 2-3x faster |

## Implementation Details

### Speed Calculation

When SHIFT is held, the camera speed is multiplied:

```
effectiveSpeed = basePanSpeed * (isShiftHeld ? shiftSpeedMultiplier : 1)
```

With default values:
- Normal speed: 5 pixels/frame
- SHIFT speed: 12.5 pixels/frame (2.5x faster)

### Diagonal Movement

Diagonal movement is normalized to prevent faster movement when pressing two keys:

```
if (moving diagonally) {
  speed /= √2
}
```

This ensures consistent movement speed in all directions.

### Performance

- Uses `requestAnimationFrame` for smooth 60fps updates
- Only updates when keys are pressed
- Efficient Set-based key tracking
- Minimal re-renders with proper state management

## Debug Mode

In development mode, a debug overlay shows:
- Current camera position (x, y)
- Current speed mode (NORMAL/FAST)
- Speed multiplier when SHIFT is active

## Browser Compatibility

- Modern browsers with ES6+ support
- Requires React 18+
- Uses standard Web APIs (requestAnimationFrame, keyboard events)

## User Feedback

Implementation based on user feedback:
> "shift + arrowkeys/WASD should move faster" - Gaudio

## License

Part of Orca Engine - See main license for details.
