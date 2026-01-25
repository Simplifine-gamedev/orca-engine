# Resources - RTS Game Components

This directory contains resource-related components for the Orca RTS game.

## GoldMine Component

The `GoldMine` component displays a gold mine resource on the game map with an improved, highly visible depletion indicator.

### Features

#### 1. **Large, Readable Gold Amount Display**
- Font size: **24px** (previously too small to read)
- High contrast with dark background and gold text
- Black text shadow for visibility on any background
- Color-coded warnings:
  - 🟡 **Gold**: Healthy (>30% remaining)
  - 🟠 **Orange**: Low (10-30% remaining)
  - 🔴 **Red**: Critical (<10% remaining)

#### 2. **Progress Bar Visualization**
- Visual representation of remaining gold percentage
- Color-coded to match the gold amount
- Percentage text overlay
- Smooth animations on depletion

#### 3. **Selection Panel**
- Comprehensive info panel when mine is selected
- Shows:
  - Large remaining gold amount (28px font)
  - Maximum gold capacity
  - Depletion percentage with large progress bar
  - Warning messages for low/critical levels
  - Helpful info text
- Centered at bottom of screen for easy visibility
- Professional dark theme with gold accents

#### 4. **Visual Feedback**
- Pulsing animation when gold is critical
- Fade animation when gold is low
- High-contrast colors for accessibility
- Emojis for quick visual identification (⛏️, ⚠️)

### Usage

```tsx
import { GoldMine } from './resources/GoldMine';

function GameMap() {
  const [selectedMine, setSelectedMine] = useState(null);

  return (
    <div className="game-map">
      <GoldMine
        remainingGold={15000}
        maxGold={20000}
        position={{ x: 100, y: 150 }}
        isSelected={selectedMine === 'mine1'}
      />
    </div>
  );
}
```

### Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `remainingGold` | `number` | Yes | Current amount of gold remaining in the mine |
| `maxGold` | `number` | Yes | Maximum gold capacity of the mine |
| `position` | `{ x: number, y: number }` | Yes | Absolute position on the game map |
| `isSelected` | `boolean` | No | Whether the mine is currently selected (shows selection panel) |

### Styling

The component uses inline styles and CSS-in-JS for maximum portability. All styles are contained within the component and require no external CSS files.

### Accessibility

- High contrast text (4.5:1 ratio minimum)
- Color-coded warnings with both color AND text
- Large, readable fonts (24px+ for primary information)
- Visual animations for critical states

### User Feedback Addressed

> "didnt see the limit for the gold mine digging, the number is too small" - Haridzieko

**Solutions implemented:**
1. ✅ Increased font size from small to **24px** (over 3x larger)
2. ✅ Added progress bar for visual depletion tracking
3. ✅ Added comprehensive selection panel with **28px** font for selected mines
4. ✅ Added color-coding and animations for low/critical states
5. ✅ Improved contrast with background and text shadows

## Files

- `GoldMine.tsx` - Main gold mine component
- `GoldMine.example.tsx` - Example usage with multiple mines
- `README.md` - This documentation file

## Future Enhancements

Potential improvements for future iterations:

- Sound effects when mine reaches low/critical levels
- Particle effects for active mining
- Tooltips on hover showing mine stats
- Keyboard shortcuts for selecting mines
- Mini-map indicators for mine locations
- Mine upgrade system UI
- Historical depletion graphs

## Testing

To test the component:

1. Import and use in your game map
2. Test with different gold amounts:
   - High: 15000+ (should show gold color)
   - Low: 3000-6000 (should show orange warning)
   - Critical: <2000 (should show red with pulsing animation)
3. Click to select and verify selection panel appears
4. Verify readability on different backgrounds

## Related Issues

- **ORC-190**: Gold mine depletion number too small to read
