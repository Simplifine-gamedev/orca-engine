# RTS Formation Control - Visual Guide

## UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  RTS Formation Control Demo          Units Selected: 4 / 12         │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                       │
│  Controls    │                 Game Viewport                        │
│              │                                                       │
│ ┌──────────┐ │   ┌─────────────────────────────────────────────┐  │
│ │Formation │ │   │ ╔═══════════════════════════════════════════╗ │  │
│ │  Type    │ │   │ ║                                           ║ │  │
│ │          │ │   │ ║    ●  ●  ●  ●                           ║ │  │
│ │  [None]  │ │   │ ║                                           ║ │  │
│ │  [Line]  │ │   │ ║    ●  ●  ●  ●                           ║ │  │
│ │  [Box]   │ │   │ ║                                           ║ │  │
│ │  [Wedge] │ │   │ ║    ●  ●  ●  ●                           ║ │  │
│ └──────────┘ │   │ ║                                           ║ │  │
│              │   │ ║              ◉ ─────> Target              ║ │  │
│ ┌──────────┐ │   │ ║              ◉ ─────>                    ║ │  │
│ │ Spread   │ │   │ ║              ◉ ─────>                    ║ │  │
│ │          │ │   │ ║              ◉ ─────>                    ║ │  │
│ │ [Tight]  │ │   │ ║         (Selected units with paths)       ║ │  │
│ │ [Normal] │ │   │ ║                                           ║ │  │
│ │ [Loose]  │ │   │ ╚═══════════════════════════════════════════╝ │  │
│ └──────────┘ │   └─────────────────────────────────────────────┘  │
│              │          (800x600 SVG Canvas with Grid)             │
│ ┌──────────┐ │                                                     │
│ │  Paths   │ │                                                     │
│ │          │ │                                                     │
│ │ [✓]Indiv.│ │                                                     │
│ │ [✓]Group │ │                                                     │
│ └──────────┘ │                                                     │
│              │                                                     │
│ ┌──────────┐ │                                                     │
│ │ Controls │ │                                                     │
│ │ (Help)   │ │                                                     │
│ └──────────┘ │                                                     │
└──────────────┴─────────────────────────────────────────────────────┘
```

---

## Formation Examples

### Line Formation
Units form a horizontal line perpendicular to facing direction:

```
Facing Direction: →

        ●
        ●
    →   ●   →  (Moving this way)
        ●
        ●
```

### Box Formation
Units arrange in a rectangular grid:

```
Facing Direction: →

    ● ● ●
→   ● ● ●   →  (Formation moves together)
    ● ● ●
```

### Wedge Formation
Triangular formation pointing forward:

```
Facing Direction: →

        ●
      ● ● ●
→   ● ● ● ●   →  (Tip points in facing direction)
```

---

## Visual Elements

### Unit States

#### Unselected Unit
```
   ┌─────┐
   │ ─>  │  Gray circle with arrow
   └─────┘
```

#### Selected Unit
```
   ╔═════╗
   ║ ─>  ║  Blue circle with glowing ring
   ╚═════╝
```

### Path Visualization

#### Individual Paths (when enabled)
```
Unit 1:  ●- - - - - -> ◎ (Target)
Unit 2:  ●- - - - - -> ◎
Unit 3:  ●- - - - - -> ◎
Unit 4:  ●- - - - - -> ◎
```

#### Group Path (when enabled)
```
    ●  ●
       |
    ●  ● (Group center)
       |
       ▼  ━━━━━━━━━>  ⊙  (Group target)
    Thick orange line
```

---

## Interaction Examples

### 1. Box Selection
```
Step 1: Click and hold
   ┌─ Mouse down
   ▼
   ●  ●  ●  ●

Step 2: Drag to create selection box
   ╔═══════════╗
   ║ ●  ●  ●  ●║
   ║           ║
   ║ ●  ●  ●  ●║
   ╚═══════════╝
          ▲
          └─ Mouse up

Step 3: Units inside box are selected (blue)
   ◉  ◉  ●  ●
   
   ◉  ◉  ●  ●
   (Selected)  (Not selected)
```

### 2. Formation Direction Drag
```
Step 1: Shift + Right-click on target location
                     ◎ <- Target click point

Step 2: Hold and drag to set direction
                     ◎
                    /
                   /  <- Yellow preview arrow
                  ●
                (Origin)

Step 3: Release - units move in formation
   
   Before:              After:
   ●  ●  ●             ●
                       ●
   ●  ●  ●    ─────>   ●  ──>  (Facing right)
                       ●
   ●  ●  ●             ●
```

---

## Color Legend

### Units
- **Gray (#64748b)**: Unselected units
- **Blue (#3b82f6)**: Selected units
- **Light Blue (#93c5fd)**: Selection ring and direction indicator

### Paths
- **Green (#4ade80)**: Individual paths for selected units
- **Gray (#94a3b8)**: Individual paths for unselected units
- **Orange (#f59e0b)**: Group path

### UI Elements
- **Yellow (#fbbf24)**: Formation direction preview
- **Blue (#3b82f6)**: Selection boxes and active buttons
- **Dark Gray (#1e293b)**: Background and panels

---

## Animation States

### 1. Idle
```
  ●
  │  Gentle rotation of direction arrow
  ●  (Future enhancement)
```

### 2. Moving
```
  ●- - - ->
     ^
     └─ Moving along path at 100px/s
```

### 3. Arriving
```
  ●──●
     Position snaps to target when within 2px
```

---

## Spread Comparison

### Tight (0.5x = 30px spacing)
```
●●●●●
●●●●●
```

### Normal (1.0x = 60px spacing)
```
● ● ● ● ●
● ● ● ● ●
```

### Loose (2.0x = 120px spacing)
```
●   ●   ●   ●   ●
●   ●   ●   ●   ●
```

---

## Button States

### Formation Type Buttons
```
┌─────────┐     Active (Blue background)
│  Line   │  <- Selected formation
└─────────┘

┌─────────┐     Inactive (Gray background)
│  Box    │  <- Hover shows darker gray
└─────────┘
```

### Checkbox Controls
```
☑ Show Individual Paths  <- Checked (visible)
☐ Show Group Path        <- Unchecked (hidden)
```

---

## Responsive Behavior

### Small Groups (1-3 units)
```
Line:    ●    or   ●  ●  ●
Box:     ●    or   ● ●
Wedge:   ●    or    ●
                   ● ●
```

### Medium Groups (4-9 units)
```
Line:    ● ● ● ● ● ● ● ● ●

Box:     ● ● ●
         ● ● ●
         ● ● ●

Wedge:      ●
          ● ● ●
        ● ● ● ●
```

### Large Groups (10+ units)
```
Line:    ● ● ● ● ● ● ● ● ● ● ● ●

Box:     ● ● ● ●
         ● ● ● ●
         ● ● ● ●

Wedge:        ●
           ● ● ●
         ● ● ● ●
       ● ● ● ● ●
```

---

## Tips for Users

1. **Quick Formation Switch**
   - Select units, choose formation, right-click to move
   - Formation applies immediately to movement

2. **Precise Facing Control**
   - Use longer drags for more precise angle control
   - Preview arrow shows exact facing before release

3. **Path Management**
   - Disable individual paths when commanding large groups
   - Enable group path to see overall movement direction

4. **Selection Techniques**
   - Double-click area for quick group select (future)
   - Shift-click to build selection incrementally

5. **Formation Tips**
   - Line formation good for defensive positions
   - Wedge formation good for charges
   - Box formation good for balanced approach

---

## Performance Notes

- Smooth 60 FPS animation on modern browsers
- Tested with up to 100 units (current demo has 12)
- SVG rendering is efficient for this scale
- Canvas rendering recommended for 200+ units (future)

---

## Browser Compatibility

Tested and working on:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Requires:
- JavaScript ES2020
- SVG 2.0 support
- CSS Grid
- Flexbox
