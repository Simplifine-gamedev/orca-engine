# Orca RTS

A real-time strategy game demo built with React, TypeScript, and Three.js.

## Features

- Control point capture system
- Real-time ownership status
- Multiple player support
- 3D visualization

## Getting Started

### Installation

```bash
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

## Control Points

Control points are strategic locations on the map that players can capture. Each control point displays its current ownership status:

- **Neutral** (Gray): No one controls this point
- **Controlled** (Green): You control this point
- **Enemy** (Red): An enemy controls this point
- **Ally** (Blue): An ally controls this point
- **Other** (Yellow): Another player controls this point

## Bug Fixes

### [ORC-134] Captured control points now show correct status

Previously, when you captured a control point, it would still display as "enemy" instead of "controlled". This has been fixed by properly checking the `ownerId` against the `playerId` in the `ControlPoint.tsx` component.

The ownership logic now correctly identifies:
1. Points you own as "controlled" (green)
2. Points owned by enemies as "enemy" (red)
3. Neutral points as "neutral" (gray)
