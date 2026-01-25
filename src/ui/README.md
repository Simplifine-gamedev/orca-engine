# UI Components

## Minimap Component

The Minimap component displays a top-down view of the game map with unit positions.

### Fixed Issues

- **ORC-174**: Removed redundant selected units count display
  - The minimap now shows selected units with visual indicators (larger size, glow effect, brighter color) instead of displaying a redundant number count
  - Selected player units are shown in bright green with a subtle glow
  - Unselected player units are shown in darker green
  - Enemy units are shown in red
  - Neutral units are shown in yellow

### Usage

```tsx
import { Minimap } from './ui/Minimap';

const units = [
  { id: '1', x: 100, y: 150, team: 'player', selected: true },
  { id: '2', x: 200, y: 250, team: 'player', selected: true },
  { id: '3', x: 300, y: 350, team: 'enemy', selected: false },
];

<Minimap
  units={units}
  mapWidth={1000}
  mapHeight={1000}
  minimapSize={200}
  onMinimapClick={(x, y) => console.log(`Clicked at ${x}, ${y}`)}
/>
```

### Props

- `units`: Array of unit objects with position, team, and selection state
- `mapWidth`: Width of the game map in world units
- `mapHeight`: Height of the game map in world units
- `minimapSize`: Size of the minimap in pixels (default: 200)
- `onMinimapClick`: Optional callback when the minimap is clicked, receives world coordinates

### Visual Indicators

Selected units are indicated by:
1. Larger circle size (radius 4 vs 2)
2. Brighter color (#00ff00 vs #00aa00 for player units)
3. Subtle glow effect around the unit

This provides clear visual feedback without cluttering the UI with redundant numerical counts.
