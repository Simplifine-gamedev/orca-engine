# Orca RTS - Multiplayer Strategy Game

A real-time strategy game built with Orca Engine featuring a comprehensive lobby system with map selection.

## Features

### Map Selection System
- **10 Unique Map Presets** with different sizes and layouts:
  - **Small Maps** (128x128): Quick 2-player matches
  - **Medium Maps** (256x256): Balanced 3-4 player games
  - **Large Maps** (512x512): Strategic 6-player battles
  - **Huge Maps** (1024x1024): Epic 8-player warfare

### Map Layouts
- **Islands**: Archipelago gameplay with naval focus
- **Continents**: Multiple landmasses separated by water
- **Pangaea**: Single large supercontinent for land battles
- **Archipelago**: Numerous islands of varying sizes
- **Desert**: Harsh terrain with limited resources
- **Arctic**: Frozen tundra with challenging conditions

### Lobby Features
- **Real-time Multiplayer**: WebSocket-based synchronization
- **Host Controls**: Map selection, game start, player management
- **Player Ready System**: All players must ready before game starts
- **Live Chat**: In-lobby communication
- **Map Preview**: Visual preview with terrain information
- **Dynamic Player List**: Color-coded players with ready status

## Architecture

### Frontend (`src/ui/`)
- **GameLobby.tsx**: Main lobby interface with player management
- **WorldMapSelector.tsx**: Interactive map selection with filters
- **MapTypes.ts**: TypeScript type definitions for maps
- **mapPresets.ts**: Configuration for all available maps

### Backend (`server/`)
- **GameServer.js**: Express + Socket.IO server
  - Lobby creation and management
  - Real-time player synchronization
  - Map selection validation
  - Game initialization

### API Endpoints

**HTTP REST API:**
- `GET /health` - Server health check
- `GET /lobbies` - List all active lobbies
- `POST /lobby/create` - Create a new lobby

**WebSocket Events:**

*Client → Server:*
- `join_lobby` - Join an existing lobby
- `leave_lobby` - Leave current lobby
- `select_map` - Change selected map (host only)
- `toggle_ready` - Toggle ready status
- `start_game` - Start the game (host only)
- `chat_message` - Send chat message

*Server → Client:*
- `lobby_joined` - Confirmation of joining
- `lobby_updated` - Lobby state changes
- `map_selected` - Map change notification
- `player_left` - Player disconnection
- `game_starting` - 3-second countdown
- `game_started` - Game initialization data
- `chat_message` - Broadcast chat messages
- `error` - Error notifications

## Installation

### Prerequisites
- Node.js 16+
- npm 8+
- Orca Engine (for game runtime)

### Setup

1. **Install dependencies:**
```bash
cd rts-game
npm install
```

2. **Start the server:**
```bash
npm start
```

The server will run on `http://localhost:3001`

For development with auto-reload:
```bash
npm run dev
```

3. **Client Setup:**
```bash
npm run dev:client
```

## Usage

### Creating a Lobby

```javascript
// HTTP Request
POST http://localhost:3001/lobby/create
{
  "name": "My Game",
  "maxPlayers": 4,
  "hostId": "player123",
  "hostName": "PlayerName"
}
```

### Joining a Lobby

```javascript
import { io } from 'socket.io-client';

const socket = io('http://localhost:3001');

socket.emit('join_lobby', {
  lobbyId: 'lobby_123',
  playerName: 'Player1',
  playerId: 'unique_id'
});

socket.on('lobby_joined', (data) => {
  console.log('Joined lobby:', data.lobby);
  console.log('My player:', data.player);
});
```

### Selecting a Map (Host Only)

```javascript
socket.emit('select_map', {
  lobbyId: 'lobby_123',
  mapId: 'large-continents'
});
```

### Starting a Game

```javascript
socket.emit('start_game', {
  lobbyId: 'lobby_123'
});

socket.on('game_started', (gameData) => {
  console.log('Game starting with map:', gameData.mapId);
  console.log('Players:', gameData.players);
  // Initialize game with map and player data
});
```

## Map Selection UI

The `WorldMapSelector` component provides:
- Grid layout of available maps
- Size filtering (Small, Medium, Large, Huge)
- Player count filtering
- Visual map previews with color coding
- Terrain distribution display
- Resource information
- Hover effects and selection highlighting

## Configuration

### Map Presets

Edit `src/config/mapPresets.ts` to add or modify maps:

```typescript
{
  id: 'custom-map',
  name: 'My Custom Map',
  description: 'A unique map layout',
  size: 'medium',
  layout: 'islands',
  width: 256,
  height: 256,
  maxPlayers: 4,
  thumbnailPath: '/assets/maps/custom.png',
  previewColor: '#3B82F6',
  terrain: {
    water: 50,
    land: 45,
    mountains: 5,
  },
  resources: {
    high: true,
    distribution: 'balanced',
  },
}
```

### Server Configuration

Environment variables (`.env`):
```env
PORT=3001
NODE_ENV=production
```

## Integration with Orca Engine

This RTS game is designed to work with Orca Engine. The game lobby handles:
1. Player matchmaking and lobby management
2. Map selection and configuration
3. Game initialization data preparation

Once the game starts, control is handed off to the Orca Engine runtime with:
- Selected map ID
- Player list with colors and teams
- Game initialization timestamp

## Development

### File Structure
```
rts-game/
├── src/
│   ├── ui/
│   │   ├── GameLobby.tsx          # Main lobby interface
│   │   └── WorldMapSelector.tsx   # Map selection component
│   ├── types/
│   │   └── MapTypes.ts            # TypeScript definitions
│   ├── config/
│   │   └── mapPresets.ts          # Map configurations
│   └── assets/                    # Map thumbnails and images
├── server/
│   └── GameServer.js              # Backend server
├── package.json
└── README.md
```

### Adding New Features

**New Map:**
1. Add preset to `mapPresets.ts`
2. Create thumbnail image in `assets/maps/`
3. Test with different player counts

**New Lobby Feature:**
1. Add UI in `GameLobby.tsx`
2. Add socket event in `GameServer.js`
3. Update state management

## Testing

Run the test suite:
```bash
npm test
```

Test the server:
```bash
curl http://localhost:3001/health
```

## License

MIT License - Open source and free to use.

## Contributing

Contributions are welcome! Please submit pull requests with:
- New map presets
- UI improvements
- Bug fixes
- Documentation updates

## Support

For issues or questions:
- GitHub Issues: [Report a bug]
- Discord: [Orca RTS Community]
- Email: support@orca-rts.example.com

---

Built with ❤️ using Orca Engine
