/**
 * Game Server - Handles multiplayer game logic and synchronization
 * Manages rally points and resource gathering on the server side
 */

const express = require('express');
const http = require('http');
const socketIO = require('socket.io');

class GameServer {
  constructor(port = 3001) {
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = socketIO(this.server, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST']
      }
    });
    this.port = port;

    // Game state
    this.gameState = {
      buildings: new Map(),
      units: new Map(),
      resources: new Map(),
      players: new Map(),
    };

    this.setupSocketHandlers();
  }

  /**
   * Setup socket.io event handlers
   */
  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`Player connected: ${socket.id}`);

      // Send current game state to new player
      socket.emit('game_state', this.serializeGameState());

      // Handle rally point setting
      socket.on('set_rally_point', (data) => {
        this.handleSetRallyPoint(socket, data);
      });

      // Handle unit spawning
      socket.on('spawn_unit', (data) => {
        this.handleSpawnUnit(socket, data);
      });

      // Handle building creation
      socket.on('create_building', (data) => {
        this.handleCreateBuilding(socket, data);
      });

      // Handle resource creation (for testing)
      socket.on('create_resource', (data) => {
        this.handleCreateResource(socket, data);
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        console.log(`Player disconnected: ${socket.id}`);
        this.gameState.players.delete(socket.id);
      });
    });
  }

  /**
   * Handle setting rally point for a building
   */
  handleSetRallyPoint(socket, data) {
    const { buildingId, position } = data;
    const building = this.gameState.buildings.get(buildingId);

    if (!building) {
      socket.emit('error', { message: 'Building not found' });
      return;
    }

    // Check if rally point is on a resource
    const resource = this.findResourceAtPosition(position);

    building.rallyPoint = {
      position,
      targetResourceId: resource?.id,
      targetResource: resource,
    };

    this.gameState.buildings.set(buildingId, building);

    // Broadcast to all players
    this.io.emit('rally_point_updated', {
      buildingId,
      rallyPoint: building.rallyPoint,
    });

    console.log(`Rally point set for building ${buildingId}`, {
      position,
      onResource: !!resource,
      resourceType: resource?.type,
    });
  }

  /**
   * Find resource at given position
   */
  findResourceAtPosition(position) {
    const RESOURCE_RADIUS = 50;

    for (const resource of this.gameState.resources.values()) {
      const distance = Math.sqrt(
        Math.pow(resource.position.x - position.x, 2) +
        Math.pow(resource.position.y - position.y, 2)
      );

      if (distance <= RESOURCE_RADIUS) {
        return resource;
      }
    }

    return null;
  }

  /**
   * Handle unit spawning from a building
   */
  handleSpawnUnit(socket, data) {
    const { buildingId, unitType = 'worker' } = data;
    const building = this.gameState.buildings.get(buildingId);

    if (!building) {
      socket.emit('error', { message: 'Building not found' });
      return;
    }

    // Create new unit
    const unit = {
      id: `unit-${Date.now()}-${Math.random()}`,
      type: unitType,
      position: { ...building.position },
      isGathering: false,
      playerId: socket.id,
    };

    // If rally point is on a resource, assign worker to gather
    if (building.rallyPoint?.targetResourceId && unitType === 'worker') {
      unit.targetResourceId = building.rallyPoint.targetResourceId;
      unit.isGathering = true;
      unit.position = { ...building.rallyPoint.position };

      console.log(`Worker spawned and assigned to gather from resource ${unit.targetResourceId}`);
    } else if (building.rallyPoint) {
      // Move to rally point
      unit.position = { ...building.rallyPoint.position };
    }

    this.gameState.units.set(unit.id, unit);

    // Broadcast to all players
    this.io.emit('unit_spawned', unit);
  }

  /**
   * Handle building creation
   */
  handleCreateBuilding(socket, data) {
    const { type, position } = data;

    const building = {
      id: `building-${Date.now()}-${Math.random()}`,
      type,
      position,
      playerId: socket.id,
      rallyPoint: null,
      spawnQueue: [],
    };

    this.gameState.buildings.set(building.id, building);

    // Broadcast to all players
    this.io.emit('building_created', building);

    console.log(`Building created: ${building.type} at (${position.x}, ${position.y})`);
  }

  /**
   * Handle resource creation
   */
  handleCreateResource(socket, data) {
    const { type, position, amount } = data;

    const resource = {
      id: `resource-${Date.now()}-${Math.random()}`,
      type,
      position,
      amount: amount || 1000,
    };

    this.gameState.resources.set(resource.id, resource);

    // Broadcast to all players
    this.io.emit('resource_created', resource);

    console.log(`Resource created: ${resource.type} at (${position.x}, ${position.y})`);
  }

  /**
   * Serialize game state for transmission
   */
  serializeGameState() {
    return {
      buildings: Array.from(this.gameState.buildings.values()),
      units: Array.from(this.gameState.units.values()),
      resources: Array.from(this.gameState.resources.values()),
    };
  }

  /**
   * Start the server
   */
  start() {
    this.server.listen(this.port, () => {
      console.log(`Game server running on port ${this.port}`);
      console.log('Features:');
      console.log('  - Rally point to resource support');
      console.log('  - Automatic worker assignment to resources');
      console.log('  - Real-time multiplayer synchronization');
    });
  }

  /**
   * Initialize some test data
   */
  initializeTestData() {
    // Create some test resources
    const goldMine = {
      id: 'resource-gold-1',
      type: 'gold',
      position: { x: 300, y: 200 },
      amount: 5000,
    };

    const woodPile = {
      id: 'resource-wood-1',
      type: 'wood',
      position: { x: 500, y: 300 },
      amount: 3000,
    };

    this.gameState.resources.set(goldMine.id, goldMine);
    this.gameState.resources.set(woodPile.id, woodPile);

    console.log('Test resources initialized');
  }
}

// Create and start server
const gameServer = new GameServer(process.env.PORT || 3001);
gameServer.initializeTestData();
gameServer.start();

module.exports = GameServer;
