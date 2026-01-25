const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const { v4: uuidv4 } = require('uuid');

class GameServer {
  constructor(port = 3001) {
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = socketIo(this.server, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST'],
      },
    });
    this.port = port;
    this.lobbies = new Map();
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupSocketHandlers();
  }

  setupMiddleware() {
    this.app.use(express.json());
    this.app.use((req, res, next) => {
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
      next();
    });
  }

  setupRoutes() {
    this.app.get('/health', (req, res) => {
      res.json({ status: 'ok', lobbies: this.lobbies.size });
    });

    this.app.post('/api/lobbies/create', (req, res) => {
      const lobbyId = uuidv4();
      const { hostName } = req.body;
      
      const lobby = {
        id: lobbyId,
        hostId: null,
        players: [],
        selectedMapId: null,
        settings: {
          maxPlayers: 4,
          startingResources: 'normal',
          gameSpeed: 'normal',
          fogOfWar: true,
        },
        createdAt: Date.now(),
      };
      
      this.lobbies.set(lobbyId, lobby);
      
      res.json({
        lobbyId,
        lobby,
      });
    });

    this.app.get('/api/lobbies/:lobbyId', (req, res) => {
      const { lobbyId } = req.params;
      const lobby = this.lobbies.get(lobbyId);
      
      if (!lobby) {
        return res.status(404).json({ error: 'Lobby not found' });
      }
      
      res.json({ lobby });
    });

    this.app.get('/api/lobbies', (req, res) => {
      const lobbiesArray = Array.from(this.lobbies.entries()).map(([id, lobby]) => ({
        id,
        playerCount: lobby.players.length,
        maxPlayers: lobby.settings.maxPlayers,
        mapSelected: !!lobby.selectedMapId,
      }));
      
      res.json({ lobbies: lobbiesArray });
    });
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`Client connected: ${socket.id}`);

      socket.on('join-lobby', ({ lobbyId, playerName }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby) {
          socket.emit('error', { message: 'Lobby not found' });
          return;
        }

        if (lobby.players.length >= lobby.settings.maxPlayers) {
          socket.emit('error', { message: 'Lobby is full' });
          return;
        }

        const player = {
          id: socket.id,
          name: playerName || `Player ${lobby.players.length + 1}`,
          team: 1,
          ready: false,
          isHost: lobby.players.length === 0,
        };

        if (player.isHost) {
          lobby.hostId = socket.id;
        }

        lobby.players.push(player);
        socket.join(lobbyId);
        socket.lobbyId = lobbyId;

        this.io.to(lobbyId).emit('lobby-update', {
          players: lobby.players,
          selectedMapId: lobby.selectedMapId,
          settings: lobby.settings,
        });

        socket.emit('joined-lobby', {
          lobbyId,
          playerId: socket.id,
          isHost: player.isHost,
        });

        console.log(`${playerName} joined lobby ${lobbyId}`);
      });

      socket.on('select-map', ({ lobbyId, mapId }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby) {
          socket.emit('error', { message: 'Lobby not found' });
          return;
        }

        if (socket.id !== lobby.hostId) {
          socket.emit('error', { message: 'Only host can select map' });
          return;
        }

        lobby.selectedMapId = mapId;
        this.io.to(lobbyId).emit('map-selected', { mapId });
        
        console.log(`Map ${mapId} selected in lobby ${lobbyId}`);
      });

      socket.on('toggle-ready', ({ lobbyId }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby) return;

        const player = lobby.players.find((p) => p.id === socket.id);
        if (player && !player.isHost) {
          player.ready = !player.ready;
          this.io.to(lobbyId).emit('lobby-update', {
            players: lobby.players,
            selectedMapId: lobby.selectedMapId,
            settings: lobby.settings,
          });
        }
      });

      socket.on('update-team', ({ lobbyId, playerId, team }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby || socket.id !== lobby.hostId) return;

        const player = lobby.players.find((p) => p.id === playerId);
        if (player) {
          player.team = team;
          this.io.to(lobbyId).emit('lobby-update', {
            players: lobby.players,
            selectedMapId: lobby.selectedMapId,
            settings: lobby.settings,
          });
        }
      });

      socket.on('update-settings', ({ lobbyId, settings }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby || socket.id !== lobby.hostId) return;

        lobby.settings = { ...lobby.settings, ...settings };
        this.io.to(lobbyId).emit('settings-updated', { settings: lobby.settings });
      });

      socket.on('chat-message', ({ lobbyId, message }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby) return;

        const player = lobby.players.find((p) => p.id === socket.id);
        if (player) {
          this.io.to(lobbyId).emit('chat-message', {
            playerId: socket.id,
            playerName: player.name,
            message,
            timestamp: Date.now(),
          });
        }
      });

      socket.on('start-game', ({ lobbyId }) => {
        const lobby = this.lobbies.get(lobbyId);
        
        if (!lobby || socket.id !== lobby.hostId) {
          socket.emit('error', { message: 'Only host can start game' });
          return;
        }

        if (!lobby.selectedMapId) {
          socket.emit('error', { message: 'No map selected' });
          return;
        }

        const allReady = lobby.players.every((p) => p.ready || p.isHost);
        if (!allReady) {
          socket.emit('error', { message: 'Not all players are ready' });
          return;
        }

        if (lobby.players.length < 2) {
          socket.emit('error', { message: 'Need at least 2 players' });
          return;
        }

        this.io.to(lobbyId).emit('game-starting', {
          mapId: lobby.selectedMapId,
          players: lobby.players,
          settings: lobby.settings,
        });

        console.log(`Game starting in lobby ${lobbyId} with map ${lobby.selectedMapId}`);
      });

      socket.on('leave-lobby', () => {
        this.handlePlayerLeave(socket);
      });

      socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
        this.handlePlayerLeave(socket);
      });
    });
  }

  handlePlayerLeave(socket) {
    const lobbyId = socket.lobbyId;
    if (!lobbyId) return;

    const lobby = this.lobbies.get(lobbyId);
    if (!lobby) return;

    const playerIndex = lobby.players.findIndex((p) => p.id === socket.id);
    if (playerIndex === -1) return;

    const wasHost = lobby.players[playerIndex].isHost;
    lobby.players.splice(playerIndex, 1);

    if (lobby.players.length === 0) {
      this.lobbies.delete(lobbyId);
      console.log(`Lobby ${lobbyId} deleted (empty)`);
    } else {
      if (wasHost && lobby.players.length > 0) {
        lobby.players[0].isHost = true;
        lobby.hostId = lobby.players[0].id;
        console.log(`New host assigned in lobby ${lobbyId}: ${lobby.players[0].name}`);
      }

      this.io.to(lobbyId).emit('lobby-update', {
        players: lobby.players,
        selectedMapId: lobby.selectedMapId,
        settings: lobby.settings,
      });
    }
  }

  start() {
    this.server.listen(this.port, () => {
      console.log(`Game server running on port ${this.port}`);
    });
  }
}

if (require.main === module) {
  const server = new GameServer(process.env.PORT || 3001);
  server.start();
}

module.exports = GameServer;
