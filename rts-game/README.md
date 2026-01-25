# Orca RTS - Idle Worker Selection Feature

A real-time strategy game with idle worker selection functionality.

## Features Implemented

### ✅ ORC-203: Button to Select Idle Workers

1. **UI Button with Idle Worker Count**
   - Shows count of idle workers in the bottom-right HUD
   - Only visible when there are idle workers
   - Eye-catching design with pulsing animation

2. **Click to Select Idle Workers**
   - Clicking the button selects all idle workers
   - Selected units have a green ring animation
   - Unit info displayed in bottom HUD

3. **Hotkey Support**
   - Press `.` (period key) to select idle workers
   - Follows Age of Empires convention
   - Works from anywhere in the game

4. **Idle Worker Indicators**
   - Idle workers have a yellow dot indicator
   - Sleep emoji badge on idle units
   - Easy visual identification

## Project Structure

```
rts-game/
├── src/
│   ├── types/
│   │   └── index.ts          # TypeScript type definitions
│   ├── store/
│   │   └── gameStore.ts      # Game state management (Zustand)
│   ├── ui/
│   │   ├── BottomHUD.tsx     # Main HUD with hotkey listener
│   │   ├── IdleWorkerButton.tsx  # Idle worker selection button
│   │   └── ResourceBar.tsx   # Resource display
│   ├── App.tsx               # Main game component
│   ├── index.tsx             # Entry point
│   └── styles.css            # Game styling
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Installation

```bash
cd rts-game
npm install
```

## Development

```bash
npm run dev
```

Open http://localhost:3000 to play the game.

## Build

```bash
npm run build
```

## How to Play

- **Click** on units to select them
- **Click** on empty space to deselect
- **Click** the idle worker button or press `.` to select all idle workers
- Watch the bottom HUD for unit information and resources

## Technologies Used

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Zustand** - State management
- **Vite** - Build tool

## Future Enhancements

- [ ] Minimap with idle worker indicators
- [ ] Unit movement on right-click
- [ ] Resource gathering mechanics
- [ ] Building construction
- [ ] Unit production
- [ ] Combat system
