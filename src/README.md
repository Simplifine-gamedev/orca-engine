# Orca RTS - Wood Gathering System

A complete wood gathering system implementation for an RTS game, featuring tree resources, worker management, and lumber camp buildings.

## Features

### 1. Wood Resource System
- **Wood as a primary resource**: Track and display wood alongside other resources (gold, stone, food)
- **Resource management**: Spend and gain wood through various game actions
- **Resource bar UI**: Real-time display of all resources at the top of the screen

### 2. Tree Management
- **Dynamic tree generation**: Automatically generates trees across the map
- **Tree states**: Trees can be full, partially harvested, or depleted
- **Tree regrowth**: Trees regrow over time (60 seconds cooldown, then gradual regrowth)
- **Visual indicators**: 
  - Tree size reflects wood amount
  - Color changes when growing (light green)
  - Transparency shows depletion status
  - Wood amount displayed above each tree

### 3. Worker System
- **Worker selection**: Click workers to select them
- **Gather assignment**: Click trees to send selected workers to gather
- **Automatic gathering**: Workers autonomously gather wood until full
- **Auto-return**: Workers automatically return to deposit when carrying capacity is full
- **Visual feedback**:
  - Blue color for idle workers
  - Gold color for gathering workers
  - White ring shows selection
  - Carrying indicator shows wood being transported

### 4. Lumber Camp Buildings
- **Strategic placement**: Build lumber camps near forests
- **Gathering bonus**: 50% faster wood gathering near lumber camps
- **Resource cost**: 100 wood + 50 gold
- **Management UI**: View and destroy lumber camps
- **Drop-off point**: Workers can return wood to lumber camps

### 5. Game Controls
- **Pause/Resume**: Control game flow
- **Game speed**: Adjust between 1x, 2x, and 3x speed
- **Real-time stats**: View tree counts and worker status

## File Structure

```
src/
├── types/
│   └── index.ts              # TypeScript type definitions
├── store/
│   ├── gameStore.ts          # Global game state (resources, workers, buildings)
│   └── treeStore.ts          # Tree management state
├── resources/
│   ├── TreeSystem.tsx        # Main tree rendering and interaction component
│   └── LumberCamp.tsx        # Lumber camp building system
├── ui/
│   └── ResourceBar.tsx       # Resource display UI component
├── App.tsx                   # Main application component
├── index.tsx                 # Application entry point
├── index.html                # HTML template
└── package.json              # Dependencies and scripts
```

## Installation

```bash
cd src
npm install
```

## Running the Game

```bash
npm run dev
```

The game will open in your browser at `http://localhost:3000`.

## How to Play

1. **Select a Worker**: Click on a blue/gold circle to select a worker
2. **Assign Gathering**: Click on a tree to send the selected worker to gather wood
3. **Watch Collection**: Workers will gather wood and automatically return when full
4. **Build Lumber Camps**: Use the sidebar to build lumber camps near forests for efficiency bonuses
5. **Manage Resources**: Keep track of your wood and other resources in the top bar
6. **Control Game**: Use pause/play and speed controls as needed

## Technical Implementation

### State Management (Zustand)
- **gameStore**: Manages global game state including resources, workers, and buildings
- **treeStore**: Handles tree generation, growth, and harvesting

### Component Architecture
- **TreeSystem**: Canvas-based rendering with click interactions
- **ResourceBar**: Real-time resource display with game controls
- **LumberCamp**: Building placement and management system

### Game Loop
- Updates every second (1000ms intervals)
- Handles tree regrowth calculations
- Processes worker gathering actions
- Manages resource deposits

### Constants (Configurable)
- `WOOD_PER_HARVEST`: 10 wood per gathering action
- `TREE_REGROWTH_TIME`: 60000ms (60 seconds) cooldown before regrowth
- `TREE_REGROWTH_RATE`: 2 wood per second during regrowth
- `LUMBER_CAMP_BONUS`: 1.5x (50% bonus) gathering speed
- `WORKER_CARRY_CAPACITY`: 10 wood maximum per trip

## Future Enhancements

- Multiple worker selection (drag selection box)
- Pathfinding for worker movement animations
- Sound effects for chopping and resource collection
- Advanced building upgrades for lumber camps
- Different tree types with varying wood amounts
- Seasonal effects on tree regrowth rates
- Worker AI for automatic resource gathering
- Minimap for large maps
- Save/load game state

## License

Part of the Orca RTS game project.
