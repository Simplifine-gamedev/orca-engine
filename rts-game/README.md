# Orca RTS - Blacksmith Building System

A real-time strategy game featuring building construction and technology research mechanics, with a focus on the Blacksmith building.

## Features

### Blacksmith Building
- **Construction**: Place and build blacksmith structures with progress tracking
- **3D Model**: Placeholder geometry (ready for GLB model integration)
- **Preview/Thumbnail**: SVG placeholder thumbnail
- **Health System**: Dynamic health bars for buildings under attack

### Research System
- **6 Research Technologies**:
  - Iron Weapons: +2 melee attack
  - Steel Armor: +1 armor for all units
  - Advanced Metallurgy: +3 attack and +2 armor for melee units
  - Weapon Sharpening: +1 attack for all units
  - Armor Reinforcement: +2 armor for all units
  - Blacksmithing Mastery: -20% cost for military units

- **Features**:
  - Tech tree with prerequisites
  - Resource costs (gold & food)
  - Research progress tracking
  - Visual feedback for locked/available/completed states

### UI Components
- **ResearchPanel**: Full-featured research interface
- **Building Component**: 3D building visualization with React Three Fiber
- **Resource Management**: Real-time resource tracking

## Project Structure

```
rts-game/
├── src/
│   ├── buildings/
│   │   ├── Building.tsx          # 3D building component
│   │   └── buildingModels.ts     # Building definitions & research data
│   ├── ui/
│   │   └── ResearchPanel.tsx     # Research UI component
│   ├── store/
│   │   ├── researchStore.ts      # Research state management (Zustand)
│   │   └── buildingStore.ts      # Building state management (Zustand)
│   └── types/
│       ├── building.ts            # Building type definitions
│       └── research.ts            # Research type definitions
├── app/
│   ├── page.tsx                  # Main game page
│   ├── layout.tsx                # App layout
│   └── globals.css               # Global styles
└── public/
    ├── thumbnails/               # Building preview images
    ├── icons/research/           # Research technology icons
    └── models/buildings/         # 3D building models (GLB)
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

Open [http://localhost:3000](http://localhost:3000) to see the game.

## Building for Production

```bash
npm run build
npm start
```

## Technologies

- **Next.js 14**: React framework
- **React Three Fiber**: 3D rendering
- **Zustand**: State management
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling

## Asset Requirements

### High Priority
1. **Blacksmith 3D Model** (`public/models/buildings/blacksmith.glb`)
   - Format: GLB
   - Poly count: 1,000-5,000 triangles
   - Scale: 1 unit = 1 meter

2. **Research Icons** (`public/icons/research/`)
   - 6 icons at 64x64px
   - Formats: PNG or SVG

3. **Building Thumbnails** (`public/thumbnails/`)
   - 128x128px or 256x256px
   - Formats: PNG or SVG

## Game Controls

- **Left Click**: Select building
- **Right Click + Drag**: Rotate camera
- **Scroll**: Zoom in/out
- **Middle Click + Drag**: Pan camera

## Testing

1. Click "Research" button to open research panel
2. Click any unlocked research technology to start research
3. Watch progress bar fill automatically
4. Complete prerequisites to unlock advanced technologies
5. Use "Add Resources" button to test resource-dependent features

## Known Issues

- 3D models not yet implemented (using placeholder geometry)
- Research icons using emoji placeholders
- No multiplayer functionality yet

## Future Enhancements

- [ ] Load actual 3D models
- [ ] Add sound effects
- [ ] Implement unit production
- [ ] Add more building types
- [ ] Multiplayer support
- [ ] Save/load game state

## License

Part of the Orca Engine project by Simplifine.
