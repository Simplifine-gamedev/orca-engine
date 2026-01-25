# Orca RTS - Resource System Demo

This is a demo implementation addressing Linear issue **ORC-160: Resource pooling unclear - players don't know what to do**.

## Problem Solved

Players were confused about the resource system and didn't understand what to do with resources. This demo implements all suggested fixes:

### ✅ Implemented Features

1. **Tutorial/Tooltips explaining resources**
   - Hover over any resource in the top bar to see detailed information
   - Each resource shows what it's used for and how to generate it
   - Building cards have comprehensive tooltips with usage tips

2. **Resource costs on building/unit buttons**
   - Every building clearly displays its cost with resource icons
   - Costs are broken down by resource type (Gold 💰, Wood 🪵, Stone 🪨, Food 🍖)

3. **Highlight when can't afford something**
   - Buildings with green borders: You can afford them
   - Buildings with red borders: Not enough resources
   - Clear badges showing "READY TO BUILD" or "CAN'T AFFORD"
   - Individual resource costs turn red when you don't have enough

4. **Resource income indicators (+X/sec)**
   - Every resource shows +X/sec income rate in the top bar
   - Buildings display their production contributions
   - Real-time resource updates every second

5. **Show what each resource is used for**
   - Detailed tooltips on each resource explaining usage
   - Building descriptions explain resource requirements
   - Info panel with gameplay instructions

## File Structure

```
rts-game/
├── src/
│   ├── game/
│   │   ├── types.ts              # Type definitions and game data
│   │   └── GameContext.tsx       # Game state management
│   ├── ui/
│   │   ├── ResourceBar.tsx       # Top resource display with tooltips
│   │   └── WorkerBuildPanel.tsx  # Building selection panel
│   └── buildings/
│       └── Building.tsx          # Individual building component
├── page.tsx                      # Main game page
└── README.md                     # This file
```

## How to Run

1. Navigate to the cloud-ide frontend directory:
   ```bash
   cd /workspace/cloud-ide/frontend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000/rts-game
   ```

## Gameplay

- Resources generate automatically every second
- Build resource-producing buildings to increase income
- Hover over anything for detailed information
- Click on buildings to see more details
- Watch your resource income grow!

## Technical Details

- Built with **Next.js 14** and **React 18**
- Uses **TypeScript** for type safety
- **Tailwind CSS** for styling
- **React Context** for state management
- Real-time resource updates with useEffect hooks

## Future Enhancements

- Unit training system
- Combat mechanics
- Multiple maps
- Multiplayer support
- Save/load game state
- More building types
- Research upgrades
- Sound effects and music

## Issue Reference

**Linear Issue:** ORC-160  
**Title:** [UI] Resource pooling unclear - players dont know what to do  
**Status:** Resolved  

All suggested fixes have been implemented and tested.
