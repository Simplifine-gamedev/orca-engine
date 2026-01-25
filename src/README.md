# Orca RTS - Marquee Selection

This is the RTS game client for Orca Engine with marquee selection support for both units and buildings.

## Features

### Implemented (ORC-205)
- ✅ **Marquee Selection Box**: Drag to select multiple entities
- ✅ **Building Selection**: Structures can be selected via marquee
- ✅ **Player Ownership Filtering**: Only player-owned entities are selectable
- ✅ **Mixed Selection**: Optional toggle for selecting units + buildings together
- ✅ **Visual Feedback**: Selected entities highlighted with gold border
- ✅ **Shift+Click**: Toggle individual entity selection

## Controls

- **Left Click**: Select single unit/building
- **Click + Drag**: Create marquee selection box
- **Shift + Click**: Toggle entity selection (add/remove from selection)
- **Checkbox**: Enable/disable mixed selection (units + buildings)

## Selection Logic

### Player Ownership
- Only entities belonging to the current player (player ID 1) can be selected
- Enemy entities (player ID 2) are visible but not selectable
- Implemented in `handleMouseDown` and `handleMouseUp` with `playerId === CURRENT_PLAYER_ID` check

### Marquee Selection
- Click and drag creates a rectangular selection box
- All entities whose bounds intersect with the box are selected
- Selection box uses dashed border animation for visibility

### Mixed Selection
- **Enabled** (default): Can select units and buildings together
- **Disabled**: Only selects the first entity type encountered in marquee

## Architecture

### Entity System
```typescript
interface Entity {
  id: string;
  type: 'unit' | 'building';
  position: Position;
  size: Size;
  playerId: number;
  selected: boolean;
  name: string;
}
```

### Selection Methods
- `isPointInEntity()`: Check if click is inside entity bounds
- `isEntityInMarquee()`: Check if entity intersects with marquee box
- `handleMouseDown()`: Start marquee or single-click selection
- `handleMouseMove()`: Update marquee box while dragging
- `handleMouseUp()`: Complete selection and apply to entities

## Development

### Setup
```bash
cd src
npm install
npm run dev
```

### Build
```bash
npm run build
```

## Demo Entities

### Player 1 (Blue/Green - Controllable)
- 5 Workers (units)
- Command Center (large building)
- Barracks (medium building)
- Supply Depot (small building)

### Player 2 (Red/Orange - Enemy)
- 3 Enemy units
- Enemy Base (building)

## Future Enhancements

- Double-click to select all units/buildings of same type
- Control groups (Ctrl+1-9)
- Selection persistence across frames
- Entity movement/commands
- Minimap with selection preview
