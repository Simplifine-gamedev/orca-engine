# Orca RTS

A simple real-time strategy game demo built with React and TypeScript, featuring floating damage numbers.

## Features

- Basic RTS gameplay with units and combat
- Floating damage numbers that appear when units deal damage
- Different colors for different damage types (physical, magic, critical)
- Smooth animations with fade out effects
- Toggle option to show/hide damage numbers

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Controls

- Click to select units
- Right-click to move or attack
- Space to toggle damage numbers on/off

## Architecture

- `src/store/gameStore.ts` - Zustand store managing game state and combat events
- `src/effects/DamageNumber.tsx` - Floating damage number component
- `src/components/` - Game UI components
- `src/types/` - TypeScript type definitions
