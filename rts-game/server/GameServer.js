const WebSocket = require('ws');

class GameServer {
  constructor(port = 8080) {
    this.wss = new WebSocket.Server({ port });
    this.gameState = {
      buildings: new Map(),
      units: new Map(),
      resources: new Map(),
      players: new Map(),
    };
    
    this.setupServer();
    console.log(`Game server started on port ${port}`);
  }

  setupServer() {
    this.wss.on('connection', (ws) => {
      console.log('Client connected');

      ws.on('message', (message) => {
        this.handleMessage(ws, message);
      });

      ws.on('close', () => {
        console.log('Client disconnected');
      });

      this.sendGameState(ws);
    });
  }

  handleMessage(ws, message) {
    try {
      const data = JSON.parse(message);
      
      switch (data.type) {
        case 'SET_RALLY_POINT':
          this.handleSetRallyPoint(data.payload);
          break;
        case 'SPAWN_UNIT':
          this.handleSpawnUnit(data.payload);
          break;
        case 'ADD_BUILDING':
          this.handleAddBuilding(data.payload);
          break;
        case 'ADD_RESOURCE':
          this.handleAddResource(data.payload);
          break;
        default:
          console.log('Unknown message type:', data.type);
      }

      this.broadcastGameState();
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  handleSetRallyPoint({ buildingId, position }) {
    const building = this.gameState.buildings.get(buildingId);
    if (!building) return;

    const detectedResource = this.detectResourceAtPosition(position);
    
    building.rallyPoint = {
      position,
      targetResource: detectedResource || undefined,
      isResourceRallyPoint: detectedResource !== null,
    };

    this.gameState.buildings.set(buildingId, building);

    console.log(
      detectedResource
        ? `Rally point set on ${detectedResource.type} resource at (${position.x}, ${position.y})`
        : `Rally point set at (${position.x}, ${position.y})`
    );
  }

  handleSpawnUnit({ buildingId, unitType }) {
    const building = this.gameState.buildings.get(buildingId);
    if (!building) return;

    const newUnit = {
      id: `unit_${Date.now()}_${Math.random()}`,
      type: unitType,
      position: { ...building.position },
      isGathering: false,
      ownerId: building.ownerId,
    };

    if (building.rallyPoint) {
      newUnit.position = { ...building.rallyPoint.position };

      if (building.rallyPoint.isResourceRallyPoint && building.rallyPoint.targetResource) {
        newUnit.isGathering = true;
        newUnit.targetResource = building.rallyPoint.targetResource.id;
        
        console.log(
          `Worker ${newUnit.id} spawned and assigned to gather from ${building.rallyPoint.targetResource.type}`
        );
      } else {
        console.log(`Unit ${newUnit.id} spawned at rally point`);
      }
    }

    this.gameState.units.set(newUnit.id, newUnit);
  }

  handleAddBuilding(building) {
    this.gameState.buildings.set(building.id, building);
    console.log(`Building ${building.id} added`);
  }

  handleAddResource(resource) {
    this.gameState.resources.set(resource.id, resource);
    console.log(`Resource ${resource.id} added`);
  }

  detectResourceAtPosition(position, radius = 50) {
    for (const [_, resource] of this.gameState.resources) {
      const distance = Math.sqrt(
        Math.pow(resource.position.x - position.x, 2) +
        Math.pow(resource.position.y - position.y, 2)
      );
      
      if (distance <= radius) {
        return resource;
      }
    }
    
    return null;
  }

  sendGameState(ws) {
    const state = {
      type: 'GAME_STATE',
      payload: {
        buildings: Array.from(this.gameState.buildings.entries()),
        units: Array.from(this.gameState.units.entries()),
        resources: Array.from(this.gameState.resources.entries()),
        players: Array.from(this.gameState.players.entries()),
      },
    };

    ws.send(JSON.stringify(state));
  }

  broadcastGameState() {
    this.wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        this.sendGameState(client);
      }
    });
  }
}

if (require.main === module) {
  new GameServer();
}

module.exports = GameServer;
