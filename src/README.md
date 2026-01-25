# Orca RTS - Attack Targeting Visual Feedback

This is a React-based RTS game demo showcasing visual cursor feedback for attack targeting.

## Features Implemented

### 1. Dynamic Cursor Changes
- **Enemy Units**: Red crosshair cursor (⚔️) indicating attackable enemy
- **Neutral Mobs**: Orange star cursor indicating attackable neutral units
- **Friendly Units**: Green circle cursor indicating friendly selection
- **Empty Space**: Blue cross cursor for move commands

### 2. Hover Highlighting
- Units glow and scale up slightly on hover
- **Enemy units**: Red outline and glow effect
- **Friendly units**: Green outline and glow effect
- **Neutral mobs**: Orange outline and glow effect

### 3. Unit System
- **RTSUnit component**: Handles friendly and enemy units
- **NeutralMob component**: Specialized component for neutral creatures
- Health bars with color coding
- Unit names displayed on hover

## File Structure

```
src/
├── App.tsx                 # Main app with cursor logic
├── types.ts               # TypeScript type definitions
├── units/
│   ├── RTSUnit.tsx       # Friendly/Enemy unit component
│   └── NeutralMob.tsx    # Neutral mob component
├── styles/
│   ├── cursor.css        # Custom cursor styles
│   └── main.css          # Main styles and hover effects
├── main.tsx              # App entry point
├── index.html            # HTML template
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
└── vite.config.ts        # Vite config
```

## How to Run

```bash
cd src
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## Controls

- **Click friendly unit** to select it
- **Click enemy or neutral** while unit selected to attack
- **Click ground** while unit selected to move
- **Hover over units** to see cursor and highlight feedback

## Implementation Details

### Cursor System
The cursor system uses CSS custom cursors with embedded SVG data URIs. The cursor changes based on the hovered unit type:
- `cursor-attack-enemy`: Red targeting reticle
- `cursor-attack-neutral`: Orange targeting star
- `cursor-friendly`: Green selection circle
- `cursor-move`: Blue movement indicator

### Hover System
Both RTSUnit and NeutralMob components implement:
- `onMouseEnter`: Triggers cursor change and highlight
- `onMouseLeave`: Resets cursor and removes highlight
- CSS transitions for smooth visual feedback

The App component maintains the cursor state and applies the appropriate CSS class to the game canvas.
