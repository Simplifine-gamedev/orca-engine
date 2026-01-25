/**
 * GameServer for Orca RTS
 * Handles lobby management, map selection, and game initialization
 */

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

class GameServer {
  constructor(port = 3001) {
    this.port = port;
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = new Server(this.server, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST'],
      },
    });

    this.lobbies = new Map();
    this.players = new Map();

    this.setupMiddleware();
    this.setupRoutes();
    this.setupSocketHandlers();
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
  }

  setupRoutes() {
    this.app.get('/health', (req, res) => {
      res.json({ status: 'ok', lobbies: this.lobbies.size, players: this.players.size });
    });

    this.app.get('/lobbies', (req, res) => {
      const lobbyList = Array.from(this.lobbies.values()).map((lobby) => ({
        id: lobby.id,
        name: lobby.name,
        playerCount: lobby.players.length,
        maxPlayers: lobby.maxPlayers,
        selectedMap: lobby.selectedMap,
        isStarted: lobby.isStarted,
      }));
      res.json(lobbyList);
    });

    this.app.post('/lobby/create', (req, res) => {
      const { name, maxPlayers = 8, hostId, hostName } = req.body;
      const lobbyId = this.generateLobbyId();

      const lobby = {
        id: lobbyId,
        name: name || `Lobby ${lobbyId}`,
        hostId,
        maxPlayers,
        players: [],
        selectedMap: 'medium-pangaea', // Default map
        isStarted: false,
        createdAt: Date.now(),
      };

      this.lobbies.set(lobbyId, lobby);
      res.json({ success: true, lobbyId, lobby });
    });
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`Player connected: ${socket.id}`);

      // Join lobby
      socket.on('join_lobby', ({ lobbyId, playerName, playerId }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby) {
          socket.emit('error', { message: 'Lobby not found' });
          return;
        }

        if (lobby.players.length >= lobby.maxPlayers) {
          socket.emit('error', { message: 'Lobby is full' });
          return;
        }

        if (lobby.isStarted) {
          socket.emit('error', { message: 'Game already started' });
          return;
        }

        const player = {
          id: playerId || socket.id,
          socketId: socket.id,
          name: playerName,
          isHost: lobby.players.length === 0,
          isReady: false,
          team: null,
          color: this.generatePlayerColor(lobby.players.length),
        };

        lobby.players.push(player);
        this.players.set(socket.id, { ...player, lobbyId });

        socket.join(lobbyId);
        socket.emit('lobby_joined', { lobby, player });
        this.io.to(lobbyId).emit('lobby_updated', lobby);

        console.log(`Player ${playerName} joined lobby ${lobbyId}`);
      });

      // Leave lobby
      socket.on('leave_lobby', () => {
        this.handlePlayerLeave(socket);
      });

      // Select map (host only)
      socket.on('select_map', ({ lobbyId, mapId }) => {
        const lobby = this.lobbies.get(lobbyId);
        const player = this.players.get(socket.id);

        if (!lobby || !player) {
          socket.emit('error', { message: 'Invalid lobby or player' });
          return;
        }

        if (player.id !== lobby.hostId && !player.isHost) {
          socket.emit('error', { message: 'Only the host can change the map' });
          return;
        }

        lobby.selectedMap = mapId;
        this.io.to(lobbyId).emit('map_selected', { mapId, selectedBy: player.name });
        this.io.to(lobbyId).emit('lobby_updated', lobby);

        console.log(`Map changed to ${mapId} in lobby ${lobbyId}`);
      });

      // Toggle ready status
      socket.on('toggle_ready', ({ lobbyId }) => {
        const lobby = this.lobbies.get(lobbyId);
        const player = this.players.get(socket.id);

        if (!lobby || !player) {
          socket.emit('error', { message: 'Invalid lobby or player' });
          return;
        }

        const lobbyPlayer = lobby.players.find((p) => p.id === player.id);
        if (lobbyPlayer && !lobbyPlayer.isHost) {
          lobbyPlayer.isReady = !lobbyPlayer.isReady;
          this.io.to(lobbyId).emit('lobby_updated', lobby);
          console.log(`Player ${player.name} ready status: ${lobbyPlayer.isReady}`);
        }
      });

      // Start game (host only)
      socket.on('start_game', ({ lobbyId }) => {
        const lobby = this.lobbies.get(lobbyId);
        const player = this.players.get(socket.id);

        if (!lobby || !player) {
          socket.emit('error', { message: 'Invalid lobby or player' });
          return;
        }

        if (player.id !== lobby.hostId && !player.isHost) {
          socket.emit('error', { message: 'Only the host can start the game' });
          return;
        }

        if (lobby.players.length < 2) {
          socket.emit('error', { message: 'Need at least 2 players to start' });
          return;
        }

        const allReady = lobby.players.every((p) => p.isReady || p.isHost);
        if (!allReady) {
          socket.emit('error', { message: 'All players must be ready' });
          return;
        }

        lobby.isStarted = true;
        
        // Prepare game initialization data
        const gameData = {
          mapId: lobby.selectedMap,
          players: lobby.players.map((p) => ({
            id: p.id,
            name: p.name,
            color: p.color,
            team: p.team,
          })),
          startTime: Date.now(),
        };

        this.io.to(lobbyId).emit('game_starting', gameData);
        
        setTimeout(() => {
          this.io.to(lobbyId).emit('game_started', gameData);
          console.log(`Game started in lobby ${lobbyId} with map ${lobby.selectedMap}`);
        }, 3000);
      });

      // Chat message
      socket.on('chat_message', ({ lobbyId, message }) => {
        const player = this.players.get(socket.id);
        if (player && lobbyId) {
          this.io.to(lobbyId).emit('chat_message', {
            player: player.name,
            message,
            timestamp: Date.now(),
          });
        }
      });

      // Disconnect
      socket.on('disconnect', () => {
        console.log(`Player disconnected: ${socket.id}`);
        this.handlePlayerLeave(socket);
      });
    });
  }

  handlePlayerLeave(socket) {
    const player = this.players.get(socket.id);
    
    if (!player) return;

    const lobby = this.lobbies.get(player.lobbyId);
    
    if (lobby) {
      lobby.players = lobby.players.filter((p) => p.socketId !== socket.id);
      
      // If host left, assign new host
      if (player.isHost && lobby.players.length > 0) {
        lobby.players[0].isHost = true;
        lobby.hostId = lobby.players[0].id;
      }

      // Delete lobby if empty
      if (lobby.players.length === 0) {
        this.lobbies.delete(player.lobbyId);
        console.log(`Lobby ${player.lobbyId} closed (empty)`);
      } else {
        this.io.to(player.lobbyId).emit('player_left', {
          playerId: player.id,
          playerName: player.name,
        });
        this.io.to(player.lobbyId).emit('lobby_updated', lobby);
      }
    }

    this.players.delete(socket.id);
    socket.leave(player.lobbyId);
  }

  generateLobbyId() {
    return `lobby_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generatePlayerColor(index) {
    const colors = [
      '#3B82F6', // Blue
      '#EF4444', // Red
      '#10B981', // Green
      '#F59E0B', // Yellow
      '#8B5CF6', // Purple
      '#EC4899', // Pink
      '#14B8A6', // Teal
      '#F97316', // Orange
    ];
    return colors[index % colors.length];
  }

  start() {
    this.server.listen(this.port, () => {
      console.log(`🎮 Orca RTS Game Server running on port ${this.port}`);
      console.log(`WebSocket server ready for connections`);
    });
  }

  stop() {
    this.server.close(() => {
      console.log('Server stopped');
    });
  }
}

// Export for use in other files
module.exports = GameServer;

// Run server if this file is executed directly
if (require.main === module) {
  const server = new GameServer(process.env.PORT || 3001);
  server.start();

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    server.stop();
  });

  process.on('SIGINT', () => {
    console.log('SIGINT signal received: closing HTTP server');
    server.stop();
    process.exit(0);
  });
}
