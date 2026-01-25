# Orca RTS Game - Map Selection & Lobby System

This directory contains the multiplayer RTS game built using the Orca Engine, featuring a comprehensive lobby and map selection system.

## Features Implemented

### 1. Multiple Map Presets ✅
- **6 Unique Maps** with different themes:
  - Coastal Bay (Easy, 4 players)
  - Desert Valley (Medium, 2 players)
  - Frozen Wastes (Hard, 6 players)
  - Volcanic Crater (Hard, 4 players)
  - Green Highlands (Easy, 3 players)
  - Urban Ruins (Medium, 8 players)

- Each map includes:
  - Unique terrain type
  - Size specifications
  - Max player count
  - Difficulty rating
  - Layout type (symmetrical, mirrored, etc.)

### 2. Map Selection UI ✅
- **WorldMapSelector Component** (`src/ui/WorldMapSelector.tsx`):
  - Grid view of all available maps
  - Filter by difficulty (Easy/Medium/Hard)
  - Sort by name, size, or player count
  - Visual map cards with thumbnails
  - Selected map highlighting
  - Terrain-specific emoji icons

### 3. Game Lobby System ✅
- **GameLobby Component** (`src/ui/GameLobby.tsx`):
  - Player management (view all players in lobby)
  - Host/player roles
  - Ready status system
  - Team assignment (4 teams)
  - In-lobby chat
  - Game settings configuration
  - Map preview in lobby
  - Start game validation

### 4. Server Implementation ✅
- **GameServer** (`src/server/GameServer.js`):
  - WebSocket-based multiplayer
  - Lobby creation and management
  - Real-time player synchronization
  - Map selection handling
  - Chat message broadcasting
  - Host migration on disconnect
  - Game start coordination

## Architecture

```
src/
├── types/
│   └── maps.ts           # Map data types and presets
├── ui/
│   ├── GameLobby.tsx     # Main lobby interface
│   └── WorldMapSelector.tsx  # Map selection UI
├── server/
│   └── GameServer.js     # Backend game server
└── assets/
    └── maps/             # Map thumbnail images
```

## Getting Started

### Installation

```bash
cd src
npm install
```

### Running the Game

Start both the frontend and server:

```bash
npm run dev:all
```

Or run them separately:

```bash
# Terminal 1 - Frontend
npm run dev

# Terminal 2 - Backend
npm run server
```

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_SERVER_URL=http://localhost:3001
```

## Usage

### Creating a Lobby

1. User clicks "Create Lobby"
2. Server generates unique lobby ID
3. User becomes host
4. Lobby URL can be shared with other players

### Selecting a Map

1. Host clicks "Select Map" button
2. Map selector displays all available maps
3. Host can filter by difficulty and sort
4. Click on a map card to select it
5. Selected map appears in lobby preview

### Starting a Game

Requirements:
- Host must select a map
- At least 2 players in lobby
- All non-host players must be ready
- Host clicks "Start Game"

### Player Actions

**As Host:**
- Select/change map
- Assign player teams
- Configure game settings
- Start game when ready

**As Player:**
- Toggle ready status
- Chat with other players
- View selected map
- Wait for game to start

## Map Configuration

Each map preset includes:

```typescript
interface MapPreset {
  id: string;              // Unique identifier
  name: string;            // Display name
  description: string;     // Map description
  thumbnail: string;       // Image path
  size: {
    width: number;         // Map width
    height: number;        // Map height
  };
  maxPlayers: number;      // Maximum players
  difficulty: 'Easy' | 'Medium' | 'Hard';
  terrain: string;         // Terrain type
  layout: string;          // Layout pattern
}
```

## WebSocket Events

### Client → Server
- `join-lobby` - Join an existing lobby
- `select-map` - Select a map (host only)
- `toggle-ready` - Toggle ready status
- `update-team` - Change player team (host only)
- `update-settings` - Update game settings (host only)
- `chat-message` - Send chat message
- `start-game` - Start the game (host only)
- `leave-lobby` - Leave current lobby

### Server → Client
- `lobby-update` - Lobby state changed
- `map-selected` - Map was selected
- `settings-updated` - Settings changed
- `chat-message` - New chat message
- `game-starting` - Game is starting
- `error` - Error occurred

## Future Enhancements

- [ ] Map thumbnail generation
- [ ] Custom map editor
- [ ] Map voting system
- [ ] Player statistics
- [ ] Match history
- [ ] Ranked matchmaking
- [ ] Spectator mode
- [ ] Replay system

## Related Issues

- Linear Issue: ORC-154 - [Lobby] Map selection screen
- Priority: Low
- Status: Implemented

## Technologies Used

- **Frontend**: React, Next.js, TypeScript, TailwindCSS
- **Backend**: Node.js, Express, Socket.IO
- **Engine**: Orca Engine (Godot fork)

## Contributing

When adding new maps:
1. Add map definition to `src/types/maps.ts`
2. Create map thumbnail at 400x400px
3. Test with different player counts
4. Update this README

## License

MIT
