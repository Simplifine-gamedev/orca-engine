# RTS Combat Demo - Visual Cursor Feedback

This demo implements visual cursor feedback for attack targeting in an RTS game, addressing Linear issue ORC-129.

## Features Implemented

### 1. Dynamic Cursor Feedback
- **Default Cursor**: When hovering over empty map space
- **Attack Cursor**: Red crosshair when hovering over enemy units or neutral mobs
- **Friendly Cursor**: Blue user icon when hovering over friendly units
- **Move Cursor**: Directional arrows when a unit is selected and hovering over empty space

### 2. Unit Hover Highlights
- **RTSUnit Component**: 
  - Glowing border effect when hovered
  - Pulsing ring animation
  - Scale-up animation on hover
  - Selection ring with rotating dashed border
  - Health bar with color-coded status

- **NeutralMob Component**:
  - Enhanced glow effect on hover
  - Double pulsing rings (inner and outer)
  - Bobbing animation
  - Attackable indicator (⚔️) appears on hover
  - Distinct visual style from regular units

### 3. Visual Indicators
- Health bars showing unit status
- Selection indicators with rotating borders
- Unit type icons (🛡️ for friendly, ⚔️ for enemy, 🐲 for neutral)
- Real-time cursor type display in header

## File Structure

```
src/
├── App.tsx                 # Main app with cursor logic and state management
├── units/
│   ├── RTSUnit.tsx        # Player and enemy unit component with hover effects
│   └── NeutralMob.tsx     # Neutral mob component with enhanced hover
├── styles.css             # Complete styling including custom cursors
├── index.tsx              # React entry point
├── index.html             # HTML template
├── package.json           # Project dependencies
├── tsconfig.json          # TypeScript configuration
├── tsconfig.node.json     # Node TypeScript configuration
├── vite.config.ts         # Vite build configuration
└── README.md              # This file
```

## How It Works

### Cursor Logic (App.tsx)
1. Tracks hovered unit and selected unit state
2. Determines cursor type based on:
   - Unit type being hovered (enemy, friendly, neutral)
   - Whether a unit is currently selected
   - Whether hovering over empty map space
3. Applies appropriate CSS class to game map container

### Hover Highlighting
- Uses `onMouseEnter` and `onMouseLeave` events
- Communicates hover state up to App component
- Applies visual effects through CSS classes and inline styles
- Includes pulsing animations and glow effects

### Custom Cursors
- Implemented using SVG data URIs in CSS
- Fallback to system cursors for compatibility
- Visually distinct for each interaction type

## Installation & Running

### Prerequisites
- Node.js 16+ and npm

### Setup
```bash
cd src
npm install
npm run dev
```

The demo will open in your browser at `http://localhost:3000`

## Usage Instructions

1. **Select a unit**: Click on a blue friendly unit to select it
2. **Hover over enemies**: Move your cursor over red enemy units or yellow neutral mobs to see the attack cursor
3. **Hover over friendlies**: Move cursor over blue units to see the friendly cursor
4. **Move command**: With a unit selected, click on empty map space (move cursor will appear)
5. **Attack command**: With a unit selected, click on an enemy or neutral mob

## Technical Details

### Custom Cursor Implementation
Cursors are defined in `styles.css` using SVG data URIs:
- Attack cursor: Red crosshair with glow effect
- Friendly cursor: Light blue user/team icon
- Move cursor: Directional arrows

### Animation Effects
- **Pulse rings**: CSS keyframe animations for expanding/fading rings
- **Selection ring**: Rotating dashed border
- **Bobbing**: Vertical translation for neutral mobs
- **Glow**: Box-shadow effects that intensify on hover

### Performance Considerations
- Uses CSS transforms for animations (GPU-accelerated)
- Event bubbling controlled with `stopPropagation`
- Minimal re-renders through `useCallback` hooks

## Browser Compatibility

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Future Enhancements

Potential improvements:
- Add particle effects on attack
- Implement unit movement animations
- Add sound effects for hover/click
- Include formation selection (drag-to-select multiple units)
- Add unit abilities with different cursors
- Implement fog of war
- Add minimap with unit positions

## Issue Resolution

This implementation resolves Linear issue **ORC-129**: [Combat] Visual cursor feedback for attack targeting

✅ Change cursor icon when hovering over attackable target
✅ Sword/crosshair cursor for enemies
✅ Different cursor for friendly units
✅ Highlight enemy on hover
✅ Hover highlight for RTSUnit
✅ Hover highlight for NeutralMob
