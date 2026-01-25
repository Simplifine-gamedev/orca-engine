# RTS Path Visibility Demo

This demo addresses Linear issue ORC-156: "[Visual] Individual unit path lines are distracting"

## Problem Solved

When moving multiple units in an RTS game, having all individual path lines visible creates visual clutter and distraction.

## Solution Implemented

### 1. Path Visibility Controls (`src/store/gameStore.ts`)

Created a Zustand store with the following settings:

- **Show Path Lines**: Global toggle to hide/show all path lines
- **Show Only Lead Unit Path**: Display path only for the lead unit (marked with yellow indicator)
- **Show Group Destination Marker**: Display a single animated marker at the group destination
- **Path Opacity**: Adjustable opacity (0-100%)
- **Path Fade Speed**: Control how quickly paths fade out (0.1-5s)

### 2. RTSUnit Component (`src/units/RTSUnit.tsx`)

Features:
- Conditional path rendering based on settings
- Lead unit indicator (yellow pulsing dot)
- Animated path lines with arrow heads
- Automatic fade-out effect
- Selection highlighting

### 3. Group Destination Marker (`src/components/GroupDestinationMarker.tsx`)

- Animated pulsing ring
- Cross-hair target indicator
- Shows single destination point for selected group

### 4. Interactive Demo (`app/rts-demo/page.tsx`)

Controls:
- Click unit to select
- Ctrl/Cmd + Click for multi-select
- Drag to box select units
- Click canvas to move selected units
- Shift + Click to add new units

## Usage

Navigate to `/rts-demo` to see the interactive demonstration.

## Technical Details

**Stack:**
- Next.js 14 (React)
- TypeScript
- Zustand (state management)
- Tailwind CSS (styling)

**Key Features:**
- Smooth animations and transitions
- Efficient state management
- Responsive controls
- Visual feedback for all interactions

## Files Created

```
cloud-ide/frontend/
├── src/
│   ├── store/
│   │   └── gameStore.ts          # State management
│   ├── units/
│   │   └── RTSUnit.tsx            # Unit component
│   └── components/
│       └── GroupDestinationMarker.tsx  # Destination indicator
└── app/
    └── rts-demo/
        └── page.tsx               # Demo page
```

## How It Solves the Issue

1. **Option to hide path lines**: ✅ Global toggle
2. **Show single group destination marker**: ✅ Animated marker when enabled
3. **Fade path lines quickly**: ✅ Adjustable fade speed (0.1-5s)
4. **Only show path for lead unit**: ✅ Toggle to show only leader's path
5. **Settings toggle for path visibility**: ✅ Full settings panel

All 5 suggested fixes from the Linear issue have been implemented!
