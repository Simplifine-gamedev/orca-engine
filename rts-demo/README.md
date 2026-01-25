# Orca RTS Demo - Marquee Selection

A web-based RTS demo showcasing marquee selection for both friendly and enemy units.

## Features

### Marquee Selection (ORC-204)

- **Enemy Unit Selection**: Drag selection box now works for both friendly and enemy units
- **Visual Distinction**: 
  - Friendly units: Green circles with blue selection outline
  - Enemy units: Red circles with orange selection outline
  - Mixed selection: Purple marquee box
- **Info Display**: Shows health and stats for all selected units (friendly and enemy)
- **Command Restrictions**: Command buttons are disabled when enemy units are selected

### Controls

- **Click & Drag**: Create marquee selection box
- **Click Unit**: Select single unit
- **Ctrl/Cmd + Click**: Toggle unit selection (add/remove)
- **Shift + Drag**: Add units to existing selection

### Implementation Details

The selection logic in `src/App.tsx` includes:

1. **Selection Box Visual Feedback**:
   - Blue for friendly-only selection
   - Orange for enemy-only selection
   - Purple for mixed selection

2. **Unit Information Panel**:
   - Displays all selected units regardless of type
   - Shows unit name, type (friendly/enemy), and health
   - Color-coded health bars

3. **Command System**:
   - Commands enabled only for friendly units
   - Warning message when enemies are selected
   - Success message when friendly units are ready

## Development

```bash
cd rts-demo
npm install
npm run dev
```

## Building

```bash
npm run build
```

## Requirements Met

- ✅ Allow marquee to select enemy units (for info/targeting)
- ✅ Distinguish between friendly and enemy selection visually
- ✅ When enemies are selected, show their info but don't allow commands
