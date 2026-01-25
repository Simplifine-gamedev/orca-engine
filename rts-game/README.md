# Orca RTS Game - Path Visibility Solution

## Overview

This project implements a comprehensive solution to the visual distraction issue when moving multiple units in an RTS game. When multiple units are selected and moving, showing all individual path lines can be overwhelming and distracting. This implementation provides multiple configurable strategies to address this problem.

## Problem Statement (ORC-157)

When moving multiple units, all of them showing their individual paths is very distracting to players.

## Implemented Solutions

This implementation provides **all 5 suggested fixes** from the Linear issue:

### 1. ✅ Option to Hide Path Lines
- Master toggle to show/hide all path lines
- Located in the settings panel
- Immediate effect on all units

### 2. ✅ Show Single Group Destination Marker
- When multiple units are selected, show one large marker at the group's centroid destination
- Animated pulsing ring with unit count indicator
- Replaces individual path lines for cleaner visual feedback
- Mode: `group-marker`

### 3. ✅ Fade Path Lines Quickly
- Path lines appear when units receive movement commands
- Automatically fade out over configurable duration (default: 1 second)
- Provides immediate feedback without persistent visual clutter
- Mode: `fade-quick`
- Configurable fade duration: 0.5s to 5s

### 4. ✅ Only Show Path for Lead Unit
- Only the first selected unit (lead unit) displays its path
- Lead unit is highlighted with orange/amber color
- Reduces visual noise while maintaining directional feedback
- Mode: `lead-only` (default)

### 5. ✅ Settings Toggle for Path Visibility
- Comprehensive settings panel with all options
- Multiple display modes available
- Adjustable opacity and fade duration
- Persistent preferences (can be extended to localStorage)

## Path Visibility Modes

### Lead Unit Only (Default)
```typescript
pathVisibilityMode: 'lead-only'
```
- Shows path only for the first selected unit
- Other units don't display paths
- Lead unit highlighted with orange glow
- Best balance between feedback and clarity

### Group Marker
```typescript
pathVisibilityMode: 'group-marker'
```
- Hides individual unit paths
- Shows single large destination marker at group centroid
- Animated pulsing ring
- Includes unit count indicator
- Best for large group movements

### Quick Fade
```typescript
pathVisibilityMode: 'fade-quick'
```
- Shows all paths initially
- Fades out over time (configurable)
- Provides immediate feedback
- Reduces long-term clutter
- Best for dynamic gameplay

### All Paths
```typescript
pathVisibilityMode: 'all'
```
- Shows paths for all selected units
- Traditional RTS behavior
- Can be overwhelming with many units
- Useful for precise formations

### None
```typescript
pathVisibilityMode: 'none'
```
- Hides all path lines completely
- Minimal visual style
- Units still show selection state
- For players who prefer clean visuals

## Project Structure

```
rts-game/
├── src/
│   ├── units/
│   │   └── RTSUnit.tsx          # Unit component with path rendering
│   ├── store/
│   │   └── gameStore.ts         # Zustand state management
│   ├── components/
│   │   ├── GameScene.tsx        # Main 3D scene
│   │   └── PathVisibilitySettings.tsx  # Settings UI
│   ├── hooks/
│   │   └── useGroupDestination.ts  # Group marker logic
│   ├── App.tsx                  # Main app component
│   ├── index.tsx                # Entry point
│   └── index.css                # Global styles
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Technical Implementation

### State Management (Zustand)
```typescript
interface GameState {
  units: Unit[];
  pathVisibilityMode: PathVisibilityMode;
  showPathLines: boolean;
  pathFadeDuration: number;
  pathOpacity: number;
  groupDestinationMarkerEnabled: boolean;
  // ... actions
}
```

### Unit Component Features
- Conditional path rendering based on mode
- Animated fade effects
- Lead unit highlighting
- Selection state visualization
- Destination markers

### Group Destination Logic
- Calculates centroid of all selected unit destinations
- Only shown when 2+ units selected
- Animated pulsing effect
- Displays unit count

## Getting Started

### Installation
```bash
cd rts-game
npm install
```

### Development
```bash
npm run dev
```

### Build
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

## Usage Example

```typescript
import { useGameStore } from './store/gameStore';

function MyComponent() {
  const { 
    setPathVisibilityMode, 
    setPathOpacity 
  } = useGameStore();
  
  // Change to lead-only mode
  setPathVisibilityMode('lead-only');
  
  // Adjust opacity
  setPathOpacity(0.5);
}
```

## Configuration Options

### Path Visibility Mode
- **Type**: `'all' | 'lead-only' | 'group-marker' | 'none' | 'fade-quick'`
- **Default**: `'lead-only'`
- **Description**: Determines how unit paths are displayed

### Show Path Lines
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Master toggle for all path visualization

### Path Opacity
- **Type**: `number` (0.0 to 1.0)
- **Default**: `0.7`
- **Description**: Transparency level of path lines

### Path Fade Duration
- **Type**: `number` (milliseconds)
- **Default**: `1000`
- **Description**: How quickly paths fade in 'fade-quick' mode

### Group Destination Marker Enabled
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Whether to show group marker in 'group-marker' mode

## Visual Indicators

### Unit States
- **Unselected**: Gray color
- **Selected**: Blue (#4ea7fc)
- **Lead Unit**: Orange glow (#ff6600)

### Path Colors
- **Regular Unit Path**: Blue (#4ea7fc)
- **Lead Unit Path**: Orange (#ff9933)
- **Group Marker**: Blue with pulsing animation

## Performance Considerations

- Paths only rendered for selected units
- Efficient fade animations using RAF
- Group destination calculated once per frame
- Minimal re-renders using Zustand

## Future Enhancements

Potential improvements not yet implemented:
- Curved/bezier path lines
- Path smoothing
- Collision avoidance visualization
- Formation preview
- Unit speed indicators
- ETA to destination
- Path waypoints
- Custom path colors per player
- Minimap path indicators

## Testing

The project includes a demo setup that:
1. Creates 8 units in a 4x2 formation
2. Selects the first 4 units
3. Assigns destinations to demonstrate path visualization
4. Allows testing all visibility modes

## Dependencies

- **React 18**: UI framework
- **Three.js**: 3D rendering
- **@react-three/fiber**: React renderer for Three.js
- **@react-three/drei**: Three.js helpers
- **Zustand**: State management
- **Vite**: Build tool
- **TypeScript**: Type safety

## Contributing

When adding new features:
1. Maintain type safety
2. Update gameStore for new settings
3. Document in README
4. Test all visibility modes
5. Consider performance impact

## License

ISC

## Credits

Developed for Orca RTS - Linear Issue ORC-157
