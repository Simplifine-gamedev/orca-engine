# Orca RTS - Wood Gathering System

A real-time strategy game featuring an automated wood gathering system.

## Features

### ✅ Implemented (ORC-161)

1. **Wood as a resource type** - Wood is tracked in the resource bar
2. **Workers chop trees for wood** - 3 workers automatically find and chop nearby trees
3. **Trees regrow over time** - Trees regenerate 10 wood every 5 seconds
4. **Wood used for resources** - Wood is deposited at the Town Hall and added to player resources
5. **Automated gathering** - Workers have AI that handles:
   - Finding nearest available tree
   - Moving to tree
   - Chopping wood (5 wood per cycle)
   - Returning to Town Hall when carrying capacity is full (10 wood)
   - Depositing wood
   - Repeating the cycle

## Game Mechanics

### Workers
- **Carry capacity**: 10 wood
- **Chop speed**: 5 wood per cycle
- **Move speed**: 2 units per tick
- **States**: idle, moving_to_tree, chopping, returning, depositing

### Trees
- **Initial wood**: 100
- **Max capacity**: 100
- **Regrowth rate**: 10 wood every 5 seconds
- Trees show health percentage and can be depleted
- Visual indicators show when being chopped

### Buildings
- **Town Hall**: Central resource deposit point for workers

## Running the Game

```bash
cd rts-game
npm install
npm run dev
```

Open [http://localhost:3001](http://localhost:3001) to play the game.

## File Structure

```
rts-game/
├── src/
│   ├── resources/
│   │   └── TreeSystem.tsx       # Tree rendering and visuals
│   ├── store/
│   │   ├── gameStore.ts         # Global game state and resources
│   │   ├── treeStore.ts         # Tree management and regrowth
│   │   └── workerStore.ts       # Worker state management
│   ├── ui/
│   │   └── ResourceBar.tsx      # Top resource display bar
│   ├── systems/
│   │   ├── WorkerAI.tsx         # Worker AI logic
│   │   ├── WorkerRenderer.tsx   # Worker rendering
│   │   ├── BuildingRenderer.tsx # Building rendering
│   │   └── TreeRegrowth.tsx     # Tree regrowth system
│   └── types/
│       └── index.ts             # TypeScript type definitions
└── app/
    ├── page.tsx                 # Main game page
    ├── layout.tsx               # App layout
    └── globals.css              # Global styles
```

## Technologies Used

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Zustand** - State management
- **Tailwind CSS** - Styling
