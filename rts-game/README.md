# Orca RTS Game - Watchtower Vision System

A real-time strategy game featuring watchtowers with visual vision indicators.

## Features

### Watchtower Vision Indicator (ORC-137)

This implementation includes the following features as requested:

1. **Eye Icon Indicator** 👁️
   - Floating eye icon above each watchtower
   - Smooth floating animation to draw attention
   - Clearly indicates vision-providing structures

2. **Vision Radius Preview**
   - Hover over any watchtower to see its vision radius
   - Pulsing circular overlay shows exact coverage area
   - Color-coded by team (green for player, red for enemy, gray for neutral)

3. **Informative Tooltip**
   - Detailed information about the watchtower
   - Shows vision radius in pixels
   - Explains the benefit of capturing/controlling the tower
   - Context-aware text based on tower ownership

4. **Additional Features**
   - Team-based color coding
   - Smooth animations and transitions
   - Responsive design for different screen sizes
   - Clean, modern UI

## Installation

```bash
cd rts-game
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The game will be available at `http://localhost:3001`

## Build

Build for production:

```bash
npm run build
```

## Project Structure

```
rts-game/
├── src/
│   ├── objects/
│   │   ├── ControlPoint.tsx    # Main watchtower component
│   │   └── ControlPoint.css    # Component styles
│   ├── components/             # Additional game components
│   ├── assets/                 # Game assets
│   ├── App.tsx                 # Main app component
│   ├── App.css                 # App styles
│   ├── index.tsx               # Entry point
│   └── index.css               # Global styles
├── public/                     # Static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## ControlPoint Component

The `ControlPoint` component represents a watchtower in the game.

### Props

```typescript
interface ControlPointProps {
  id: string;              // Unique identifier
  x: number;               // X position on map
  y: number;               // Y position on map
  team?: 'neutral' | 'player' | 'enemy';  // Tower ownership
  visionRadius?: number;   // Vision radius in pixels (default: 150)
  isWatchtower?: boolean;  // Enable watchtower features (default: true)
}
```

### Usage Example

```tsx
import { ControlPoint } from './objects/ControlPoint';

<ControlPoint
  id="tower-1"
  x={200}
  y={300}
  team="player"
  visionRadius={180}
  isWatchtower={true}
/>
```

## Design Decisions

1. **Eye Icon**: Used the 👁️ emoji for simplicity and immediate recognition. Can be replaced with custom SVG for production.

2. **Vision Radius**: Implemented as a dashed circle that appears on hover, making it clear without cluttering the UI.

3. **Tooltip**: Positioned above the tower to avoid obstruction, with detailed information about vision benefits.

4. **Animations**: 
   - Floating animation for eye icon (draws attention)
   - Pulsing animation for vision radius (indicates active area)
   - Smooth hover transitions (professional feel)

5. **Color Scheme**:
   - Green (#4CAF50): Player-controlled
   - Red (#f44336): Enemy-controlled
   - Gray (#9E9E9E): Neutral/uncaptured

## User Feedback Implementation

Based on feedback from Haridzieko:
> "there can be an eye icon or sth on top of the watchtowers so its more intuitive and they figure out what conquering the towers do before doing it"

**Implemented Solutions:**
- ✅ Eye icon prominently displayed above watchtowers
- ✅ Vision radius becomes visible on hover (clear visual feedback)
- ✅ Tooltip explains vision benefit before capture
- ✅ Makes the purpose of watchtowers immediately clear to new players

## Future Enhancements

Potential improvements for future iterations:

1. Add custom SVG eye icon for better visual quality
2. Implement fog of war system that integrates with vision
3. Add capture progress indicator
4. Show enemy units revealed by watchtower vision
5. Add sound effects for hover and capture
6. Animate vision radius expansion when tower is captured
7. Add minimap integration showing tower vision coverage

## Technologies Used

- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe component development
- **Vite**: Fast build tool and dev server
- **CSS3**: Custom animations and styling

## License

MIT

## Contributing

This is part of the Orca RTS game project. For contributions, please follow the project's contribution guidelines.
