# Orca Cloud IDE - Frontend

This is the frontend for the Orca Cloud IDE, built with Next.js and React.

## Features

### Cloud IDE
- Monaco code editor integration
- Live 3D viewport via VNC
- Real-time collaboration via WebSocket
- Project file management

### RTS Game Demo
A demonstration of an RTS game built with React, showcasing:

#### Building System
- Multiple building types (Town Hall, Blacksmith, Barracks, etc.)
- 3D model preview system with fallback rendering
- Building thumbnails and previews
- Construction progress visualization
- Health indicators
- Resource cost display

#### Blacksmith Building
The Blacksmith is a research building that provides:
- **7 research technologies:**
  - Iron Weapons (+2 attack)
  - Steel Weapons (+3 attack, requires Iron Weapons)
  - Leather Armor (+1 armor)
  - Chain Mail (+2 armor, requires Leather Armor)
  - Plate Armor (+3 armor, requires Chain Mail)
  - Advanced Forging (reduces research time by 25%)
  - Siege Engineering (unlocks Catapult and Ballista units)

#### Research System
- Technology tree with prerequisites
- Resource cost validation
- Real-time research progress tracking
- Research panel UI component
- Multiple technology effects (stat boosts, unit unlocks, bonuses)

## Project Structure

```
src/
├── types/
│   └── game.ts              # Core game type definitions
├── buildings/
│   ├── Building.tsx         # Building component
│   ├── buildingTypes.ts     # Building type definitions
│   └── buildingModels.ts    # 3D model management
├── data/
│   └── researchTechs.ts     # Research technology definitions
├── store/
│   └── researchStore.ts     # Research state management
├── ui/
│   └── ResearchPanel.tsx    # Research UI component
└── app/
    ├── page.tsx             # Main IDE page
    └── rts-demo/
        └── page.tsx         # RTS game demo

public/
├── images/buildings/        # Building thumbnails
├── models/buildings/        # 3D building models (.glb)
└── icons/                   # Research tech icons
```

## Running the Project

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) for the IDE.
Open [http://localhost:3000/rts-demo](http://localhost:3000/rts-demo) for the RTS demo.

## Building for Production

```bash
npm run build
npm start
```

## Adding 3D Models

To add 3D models for buildings:

1. Place `.glb` files in `public/models/buildings/`
2. Place thumbnail images in `public/images/buildings/`
3. The system will automatically use these files if available
4. Falls back to generated SVG previews if files are missing

## Technologies

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Monaco Editor
- Socket.io Client
