# Orca RTS - Idle Worker Selection

This implementation adds an idle worker selection feature to the RTS game.

## Features

### 1. Idle Worker Button
- Shows count of idle workers
- Appears in the resource bar at the bottom of the screen
- Orange color for visibility
- Hover effects for better UX
- Automatically hides when no workers are idle

### 2. Hotkey Support
- Press `.` (period key) to select all idle workers
- Press `Escape` to deselect all units
- Hotkeys work anywhere except when typing in input fields

### 3. Game Store
- Centralized state management for units, selection, and resources
- Efficient selectors for idle workers
- Subscribe-based updates for React components

## File Structure

```
src/
├── types/
│   └── game.ts          # TypeScript interfaces for game entities
├── store/
│   └── gameStore.ts     # State management and game logic
├── hooks/
│   └── useHotkeys.ts    # Keyboard shortcut hook
├── ui/
│   ├── IdleWorkerButton.tsx  # Button component
│   ├── ResourceBar.tsx       # Resource display with idle button
│   └── BottomHUD.tsx         # Bottom UI container
├── App.tsx              # Main application component
├── main.tsx             # React entry point
└── index.html           # HTML entry point
```

## Usage

### Selecting Idle Workers

**Via Button:**
1. Look for the orange button in the bottom resource bar
2. The button shows the count of idle workers
3. Click the button to select all idle workers

**Via Hotkey:**
1. Press `.` (period) key at any time
2. All idle workers will be selected

### How It Works

1. **Worker State**: Each unit has an `isIdle` flag
2. **Store Selector**: `getIdleWorkers()` filters workers by `isIdle = true`
3. **Selection Action**: `selectIdleWorkers()` updates unit selection state
4. **UI Updates**: Components subscribe to store changes and re-render

## Implementation Details

### Worker Detection
Workers are considered idle when:
- `unit.type === UnitType.WORKER`
- `unit.isIdle === true`
- No active task assigned

### Hotkey System
- Uses native keyboard events
- Prevents conflicts with text input
- Easily extensible for additional hotkeys

### State Management
- Custom lightweight store (no external dependencies)
- Pub-sub pattern for reactivity
- Efficient selector-based queries

## Testing

The app initializes with 4 test workers:
- 3 idle workers (worker-1, worker-2, worker-4)
- 1 busy worker (worker-3)

## Future Enhancements

- [ ] Minimap indicator for idle workers
- [ ] Sound notification when workers become idle
- [ ] Cycle through idle workers on repeated presses
- [ ] Persistent idle worker queue
- [ ] Camera centering on selected workers

## Browser Support

Requires modern browsers with:
- ES2020+ support
- React 18+
- TypeScript 5+
