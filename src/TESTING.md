# Testing Guide - Lobby & Map Selection

This guide explains how to test the lobby and map selection features.

## Prerequisites

1. Node.js v18+ installed
2. npm or yarn package manager

## Setup

### 1. Install Dependencies

```bash
cd src
npm install

cd server
npm install
```

### 2. Start the Server

```bash
cd src/server
npm start
```

The server will start on `http://localhost:3001`

### 3. Start the Frontend (in a new terminal)

```bash
cd src
npm run dev
```

The frontend will start on `http://localhost:3000`

## Test Scenarios

### Scenario 1: Creating and Selecting Maps (Solo)

1. **Open the game**: Navigate to `http://localhost:3000`
2. **Enter name**: Type your player name
3. **Create lobby**: Click "Create New Lobby"
4. **Select map**: Click "Select a Map" or "Change Map"
5. **Browse maps**: 
   - View all 6 available maps
   - Try filtering by difficulty (Easy/Medium/Hard)
   - Try sorting by name, size, or player count
6. **Select a map**: Click on any map card
7. **Verify selection**: 
   - Selected map should have blue border
   - Map details should appear in lobby
   - ✓ Selected badge should appear on the map

### Scenario 2: Multiplayer Lobby

#### Setup
1. Open two browser windows (or use incognito mode)
2. In Window 1: Create a lobby as "Player 1"
3. Copy the Lobby ID from the URL or interface
4. In Window 2: Enter name "Player 2" and join using the Lobby ID

#### Tests

**Test 2.1: Player Sync**
- ✅ Both players should see each other in the player list
- ✅ Player 1 should have HOST badge
- ✅ Player 2 should have ready toggle button

**Test 2.2: Map Selection Sync**
- In Window 1 (host): Select a map
- ✅ Map should update in both windows
- In Window 2: Try selecting a map
- ✅ Should not be able to change (only host can)

**Test 2.3: Ready System**
- In Window 2: Click "Ready" button
- ✅ Button should change to "Not Ready"
- ✅ Player 2 should show "READY" badge in Window 1
- ✅ Host's "Start Game" button should enable

**Test 2.4: Team Assignment**
- In Window 1 (host): Change Player 2's team
- ✅ Team selection should update in both windows

**Test 2.5: Chat System**
- In Window 1: Send a message "Hello from Player 1"
- ✅ Message should appear in both windows
- In Window 2: Send a message "Hello from Player 2"
- ✅ Message should appear in both windows
- ✅ Messages should show correct player names

**Test 2.6: Game Start**
- Ensure map is selected
- Ensure Player 2 is ready
- In Window 1: Click "Start Game"
- ✅ Both windows should receive "game-starting" event
- ✅ Game should load with selected map

### Scenario 3: Host Migration

1. Create lobby with Player 1 (host)
2. Player 2 joins
3. Player 3 joins
4. Close Player 1's window (host leaves)
5. ✅ Player 2 should become the new host
6. ✅ Player 2 should now be able to select maps and start game

### Scenario 4: Settings Configuration

1. As host, modify game settings:
   - Starting Resources: Try Low/Normal/High
   - Game Speed: Try Slow/Normal/Fast
   - Fog of War: Toggle on/off
2. ✅ Settings should persist for game start

### Scenario 5: Edge Cases

**Test 5.1: Lobby Full**
1. Create lobby with max 4 players
2. Have 5 players try to join
3. ✅ 5th player should see "Lobby is full" error

**Test 5.2: Start Game Validation**
- Try starting with no map selected
  - ✅ Button should be disabled
- Try starting with only 1 player
  - ✅ Button should show "Need More Players"
- Try starting when players not ready
  - ✅ Button should show "Waiting for Players..."

**Test 5.3: Map Filtering**
- Set filter to "Hard"
  - ✅ Should show only Frozen Wastes and Volcanic Crater
- Set filter to "Easy"
  - ✅ Should show Coastal Bay and Green Highlands
- Sort by "Max Players"
  - ✅ Maps should order by player count

**Test 5.4: Disconnect Handling**
1. Have 3 players in lobby
2. One player refreshes their browser
3. ✅ Should reconnect and see updated lobby state
4. One player closes tab
5. ✅ Other players should see them removed from lobby

### Scenario 6: WebSocket Events

Use browser DevTools to monitor WebSocket frames:

**Expected Events:**

*Client → Server:*
```javascript
join-lobby
select-map
toggle-ready
update-team
update-settings
chat-message
start-game
leave-lobby
```

*Server → Client:*
```javascript
joined-lobby
lobby-update
map-selected
settings-updated
chat-message
game-starting
error
```

## Testing Checklist

- [ ] Can create a lobby
- [ ] Can join a lobby
- [ ] Can view all 6 maps
- [ ] Can filter maps by difficulty
- [ ] Can sort maps by name/size/players
- [ ] Can select a map (host only)
- [ ] Selected map shows in lobby
- [ ] Player list updates in real-time
- [ ] Ready system works correctly
- [ ] Team assignment works
- [ ] Chat messages send and receive
- [ ] Host can modify settings
- [ ] Game start validation works
- [ ] Game starts with correct config
- [ ] Host migration works
- [ ] Disconnect handling works
- [ ] Multiple lobbies can run simultaneously

## Performance Testing

### Load Test
```bash
# Install artillery if not already installed
npm install -g artillery

# Create artillery config (artillery.yml):
config:
  target: 'http://localhost:3001'
  socketio:
    transports: ['websocket']
scenarios:
  - engine: socketio
    flow:
      - emit:
          channel: 'join-lobby'
          data: 
            lobbyId: 'test-lobby'
            playerName: 'LoadTestPlayer'

# Run test
artillery run artillery.yml
```

### Expected Performance
- Server should handle 100+ concurrent connections
- Lobby updates should propagate in < 50ms
- Chat messages should have < 100ms latency
- Map selection should be instant

## Common Issues

### Issue: Can't connect to server
**Solution**: Ensure server is running on port 3001

### Issue: Maps not displaying
**Solution**: Check console for errors, ensure MAP_PRESETS is imported

### Issue: Players not syncing
**Solution**: Check WebSocket connection, ensure both clients have same lobby ID

### Issue: Ready button not working
**Solution**: Ensure player is not the host (hosts don't have ready button)

## Debugging

### Enable Verbose Logging

**Server side** (`GameServer.js`):
```javascript
// Add after socket connection
console.log('Socket events:', socket.eventNames());
```

**Client side** (`useGameLobby.ts`):
```javascript
// Add in useEffect
newSocket.onAny((event, ...args) => {
  console.log('Socket event:', event, args);
});
```

### Check Lobby State

In browser console:
```javascript
// Check current lobby state
fetch('http://localhost:3001/api/lobbies')
  .then(r => r.json())
  .then(console.log)
```

## Automated Tests

### Unit Tests (Coming Soon)
- Map filtering logic
- Sorting algorithms
- Ready state validation
- Team assignment validation

### Integration Tests (Coming Soon)
- Full lobby flow
- Multi-player scenarios
- Host migration
- Error handling

### E2E Tests (Coming Soon)
- Playwright/Cypress tests
- Full user journeys
- Cross-browser testing

## Continuous Testing

For continuous development:

```bash
# Terminal 1: Server with auto-reload
cd src/server
npm run dev

# Terminal 2: Frontend with hot-reload
cd src
npm run dev

# Terminal 3: Watch mode tests (when implemented)
npm run test:watch
```

## Reporting Issues

When reporting bugs, include:
1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Browser console errors
5. Network tab screenshots
6. Server logs

## Next Steps

After all tests pass:
1. Deploy server to production
2. Configure CORS for production domain
3. Set up monitoring/logging
4. Implement analytics
5. Add automated tests to CI/CD

---

**Last Updated**: 2026-01-25  
**Test Coverage**: Manual testing only (automated tests TBD)
