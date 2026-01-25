# Integration Guide - Gold Mine Readability Fix (ORC-123)

## Quick Start

This fix addresses the readability issue where gold mine depletion numbers were too small to read during gameplay.

## What Changed

### Visual Improvements Summary

```
BEFORE:                          AFTER:
┌─────────────┐                 ┌─────────────┐
│  [Mine]     │                 │  [Mine]     │
│   1500      │  ← Small 10px   │   1500/15000│  ← Large 18px bold
└─────────────┘                 │  ▓▓▓▓░░░░░░ │  ← Progress bar
                                └─────────────┘

Selection Panel (NEW):
┌──────────────────────────┐
│ Gold Mine                │
│                          │
│ Gold Remaining:          │
│    15,000                │  ← 28px bold
│                          │
│ Maximum Capacity:        │
│    15,000                │
│                          │
│ Depletion Status:        │
│ ▓▓▓▓▓▓▓▓░░░░  75.0%     │
│                          │
│ ⚠ Mine nearly depleted!  │  ← Warning (<25%)
└──────────────────────────┘
```

## File Structure

```
src/resources/
├── GoldMine.tsx              # Main component (primary fix)
├── GoldMine.example.tsx      # Interactive demo
├── types.ts                  # TypeScript definitions
├── README.md                 # Detailed documentation
└── INTEGRATION_GUIDE.md      # This file
```

## Integration Steps

### Step 1: Copy Files to Your Project

```bash
# Copy the resources directory to your RTS game
cp -r src/resources /path/to/your-rts-game/src/
```

### Step 2: Update Imports

In your game's resource manager or map component:

```tsx
import { GoldMine } from './resources/GoldMine';
import { GoldMineState, ResourceType } from './resources/types';
```

### Step 3: Replace Old Component

**Before:**
```tsx
<div className="mine">
  <img src="mine.png" />
  <span className="small-text">{goldAmount}</span>
</div>
```

**After:**
```tsx
<GoldMine
  goldRemaining={mine.goldRemaining}
  maxGold={mine.maxGold}
  position={mine.position}
  isSelected={selectedMineId === mine.id}
/>
```

### Step 4: Add Selection Handling

```tsx
const [selectedMine, setSelectedMine] = useState<string | null>(null);

const handleMineClick = (mineId: string) => {
  setSelectedMine(mineId);
  // Your existing selection logic...
};
```

## Dependencies

This component uses standard React and doesn't require additional dependencies:

```json
{
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.0.0",
    "typescript": "^5.0.0"
  }
}
```

Optional: For styled-jsx support (if you prefer):
```bash
npm install styled-jsx
```

## CSS Framework Compatibility

The component uses inline styles and is compatible with:
- ✅ Plain CSS
- ✅ CSS Modules
- ✅ Tailwind CSS
- ✅ Styled Components
- ✅ Emotion
- ✅ styled-jsx (currently used)

## Customization

### Adjust Font Sizes

```tsx
// In GoldMine.tsx, modify the fontSize values:
style={{
  fontSize: '20px',  // Increase from 18px if needed
  fontWeight: 'bold',
  // ...
}}
```

### Change Colors

```tsx
// Color variables for easy theming:
const GOLD_COLOR = '#ffd700';
const WARNING_COLOR = '#ff4444';
const BACKGROUND_COLOR = 'rgba(20, 20, 30, 0.95)';
```

### Adjust Warning Threshold

```tsx
// In GoldMine.tsx, change the warning percentage:
const isNearlyDepleted = depletionPercentage < 30; // Changed from 25%
```

## Performance Optimization

For games with many mines (50+), consider:

### 1. Memoization

```tsx
import React, { memo } from 'react';

export const GoldMine = memo<GoldMineProps>(
  ({ goldRemaining, maxGold, position, isSelected }) => {
    // Component code...
  },
  (prev, next) => {
    // Custom comparison for better performance
    return (
      prev.goldRemaining === next.goldRemaining &&
      prev.isSelected === next.isSelected
    );
  }
);
```

### 2. Hide Distant Mines

```tsx
// Only render detailed info for visible mines
const isVisible = checkIfVisible(mine.position, camera);
return isVisible ? <GoldMine {...props} /> : <SimpleMineSprite />;
```

### 3. Lazy Selection Panel

```tsx
// Only render selection panel when needed
{isSelected && selectedMine && (
  <Suspense fallback={<div>Loading...</div>}>
    <GoldMineSelectionPanel {...selectedMine} />
  </Suspense>
)}
```

## Testing the Fix

### Visual Regression Test

```tsx
import { render, screen } from '@testing-library/react';
import { GoldMine } from './GoldMine';

test('displays gold amount with correct font size', () => {
  render(
    <GoldMine
      goldRemaining={5000}
      maxGold={10000}
      position={{ x: 0, y: 0 }}
    />
  );
  
  const text = screen.getByText(/5000/);
  const styles = window.getComputedStyle(text);
  expect(styles.fontSize).toBe('18px');
  expect(styles.fontWeight).toBe('bold');
});
```

### User Acceptance Testing

1. ✅ Can you read the gold amount from normal gameplay distance?
2. ✅ Is the progress bar clearly visible?
3. ✅ Does the selection panel provide useful information?
4. ✅ Are warning states noticeable without being intrusive?

## Rollout Strategy

### Phase 1: A/B Testing (Optional)
- Deploy to 10% of players
- Gather feedback on readability
- Monitor performance metrics

### Phase 2: Full Rollout
- Deploy to all players
- Monitor user feedback channels
- Track resource management metrics

### Phase 3: Iteration
- Collect feedback for 1-2 weeks
- Make minor adjustments if needed
- Consider extending to other resources

## Rollback Plan

If issues arise, revert by:

```bash
git revert <commit-hash>
```

Or temporarily disable:

```tsx
// In your game config
const USE_NEW_MINE_UI = false;

{USE_NEW_MINE_UI ? (
  <GoldMine {...props} />
) : (
  <OldMineComponent {...props} />
)}
```

## Monitoring

Track these metrics post-deployment:

- **User Feedback**: Readability complaints (should decrease)
- **Performance**: Frame rate with multiple mines
- **Engagement**: Resource management efficiency
- **Errors**: Console errors related to mine rendering

## Support

For questions or issues:
- Check the [README.md](./README.md) for detailed documentation
- Review the [example file](./GoldMine.example.tsx) for usage patterns
- Open an issue in the Linear project (Orca RTS)

## Version History

- **v1.0** (2026-01-25): Initial release
  - 80% larger font size (18px)
  - Progress bar implementation
  - Selection panel with detailed stats
  - Warning system for low resources

---

**Related Issue**: ORC-123  
**User Feedback**: Haridzieko - "didn't see the limit for the gold mine digging, the number is too small"  
**Status**: ✅ Fixed and ready for integration
