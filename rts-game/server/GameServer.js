/**
 * Game Server - Server-side game logic for Orca RTS
 * 
 * PACING IMPROVEMENTS (ORC-145):
 * - Support for scout units for early exploration
 * - Reduced build times validation
 * - Faster resource gathering synchronization
 */

const EventEmitter = require('events');

class GameServer extends EventEmitter {
  constructor() {
    super();
    this.games = new Map();
    this.players = new Map();
    this.tickRate = 50; // 50ms tick rate (20 TPS)
    this.gameLoopInterval = null;
  }

  /**
   * Create a new game session
   */
  createGame(gameId, hostPlayerId) {
    const game = {
      id: gameId,
      host: hostPlayerId,
      players: new Set([hostPlayerId]),
      state: this.getInitialGameState(),
      status: 'waiting', // waiting, playing, ended
      createdAt: Date.now(),
      startedAt: null,
    };

    this.games.set(gameId, game);
    this.emit('game:created', { gameId, hostPlayerId });
    return game;
  }

  /**
   * Initial game state with improved starting resources (ORC-145)
   */
  getInitialGameState() {
    return {
      resources: {
        gold: 200,    // Increased from 100 (ORC-145)
        wood: 200,    // Increased from 100 (ORC-145)
        food: 150,    // Increased from 100 (ORC-145)
        stone: 100,   // Increased from 50 (ORC-145)
      },
      units: this.createStartingUnits(),
      buildings: [{
        id: 'town-center-0',
        type: 'town_center',
        position: { x: 200, y: 200 },
        health: 1000,
        maxHealth: 1000,
        ownerId: null,
      }],
      population: 5, // 4 villagers + 1 scout (ORC-145)
      maxPopulation: 10,
      gameTime: 0,
    };
  }

  /**
   * Create starting units - includes scout unit (ORC-145)
   */
  createStartingUnits() {
    const units = [];
    
    // 4 villagers (increased from 3)
    for (let i = 0; i < 4; i++) {
      units.push({
        id: `villager-${i}`,
        type: 'villager',
        position: { x: 100 + i * 20, y: 100 },
        health: 50,
        maxHealth: 50,
        speed: 1.2,
        gatherRate: 1.0, // Increased gathering rate (ORC-145)
        ownerId: null,
      });
    }
    
    // 1 scout for early exploration (NEW - ORC-145)
    units.push({
      id: 'scout-0',
      type: 'scout',
      position: { x: 150, y: 150 },
      health: 60,
      maxHealth: 60,
      speed: 2.5,
      gatherRate: 0.5,
      visionRadius: 150,
      ownerId: null,
    });
    
    return units;
  }

  /**
   * Unit costs with scout unit (ORC-145)
   */
  getUnitCost(unitType) {
    const costs = {
      villager: { gold: 50, food: 25 },
      scout: { gold: 40, food: 15 },     // NEW: Scout unit
      warrior: { gold: 60, food: 40 },
      archer: { gold: 45, food: 35, wood: 20 },
    };
    return costs[unitType] || null;
  }

  /**
   * Build times - reduced for early game buildings (ORC-145)
   */
  getBuildTime(buildingType) {
    const buildTimes = {
      town_center: 60,
      barracks: 15,      // Reduced from 30s (ORC-145)
      farm: 10,          // Reduced from 20s (ORC-145)
      lumber_mill: 12,   // Reduced from 25s (ORC-145)
      mining_camp: 12,   // Reduced from 25s (ORC-145)
      house: 8,          // Reduced from 15s (ORC-145)
    };
    return buildTimes[buildingType] || 30;
  }

  /**
   * Resource gathering rates - increased (ORC-145)
   */
  getGatheringRate(unitType, resourceType) {
    const rates = {
      villager: {
        gold: 1.0,   // Increased from 0.5 (ORC-145)
        wood: 1.0,   // Increased from 0.5 (ORC-145)
        food: 1.2,   // Increased from 0.5 (ORC-145)
        stone: 0.8,  // Increased from 0.5 (ORC-145)
      },
      scout: {
        gold: 0.5,
        wood: 0.5,
        food: 0.8,
        stone: 0.5,
      }
    };
    return rates[unitType]?.[resourceType] || 0.5;
  }

  /**
   * Join a game
   */
  joinGame(gameId, playerId) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    
    if (game.status !== 'waiting') {
      throw new Error('Game already started');
    }
    
    game.players.add(playerId);
    this.players.set(playerId, gameId);
    this.emit('player:joined', { gameId, playerId });
    
    return game;
  }

  /**
   * Start a game
   */
  startGame(gameId) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    
    game.status = 'playing';
    game.startedAt = Date.now();
    
    this.emit('game:started', { gameId });
    
    // Start game loop if not already running
    if (!this.gameLoopInterval) {
      this.startGameLoop();
    }
    
    return game;
  }

  /**
   * Main game loop
   */
  startGameLoop() {
    this.gameLoopInterval = setInterval(() => {
      const deltaTime = this.tickRate / 1000; // Convert to seconds
      
      this.games.forEach((game, gameId) => {
        if (game.status === 'playing') {
          this.updateGame(gameId, deltaTime);
        }
      });
    }, this.tickRate);
  }

  /**
   * Update game state
   */
  updateGame(gameId, deltaTime) {
    const game = this.games.get(gameId);
    if (!game) return;
    
    game.state.gameTime += deltaTime;
    
    // Update resource gathering
    game.state.units.forEach(unit => {
      if (unit.isGathering && unit.targetResource) {
        const gatherRate = this.getGatheringRate(unit.type, unit.targetResource);
        const amount = gatherRate * deltaTime;
        game.state.resources[unit.targetResource] += amount;
      }
    });
    
    // Update building construction
    game.state.buildings.forEach(building => {
      if (building.isBuilding) {
        building.buildProgress += (deltaTime / building.buildTime) * 100;
        if (building.buildProgress >= 100) {
          building.isBuilding = false;
          building.buildProgress = 100;
          this.emit('building:completed', { gameId, buildingId: building.id });
        }
      }
    });
    
    // Emit state update
    this.emit('game:update', { gameId, state: game.state });
  }

  /**
   * Handle player action - train unit
   */
  trainUnit(gameId, playerId, unitType, buildingId) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    
    const cost = this.getUnitCost(unitType);
    if (!cost) {
      throw new Error('Invalid unit type');
    }
    
    // Check if player has enough resources
    for (const [resource, amount] of Object.entries(cost)) {
      if (game.state.resources[resource] < amount) {
        throw new Error(`Not enough ${resource}`);
      }
    }
    
    // Check population cap
    if (game.state.population >= game.state.maxPopulation) {
      throw new Error('Population cap reached');
    }
    
    // Deduct resources
    for (const [resource, amount] of Object.entries(cost)) {
      game.state.resources[resource] -= amount;
    }
    
    // Find building
    const building = game.state.buildings.find(b => b.id === buildingId);
    if (!building) {
      throw new Error('Building not found');
    }
    
    // Create unit (will be added after build time)
    const unitId = `${unitType}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const unit = {
      id: unitId,
      type: unitType,
      position: { ...building.position },
      health: this.getUnitMaxHealth(unitType),
      maxHealth: this.getUnitMaxHealth(unitType),
      speed: this.getUnitSpeed(unitType),
      ownerId: playerId,
      isTraining: true,
      trainingProgress: 0,
      trainingTime: this.getUnitBuildTime(unitType),
    };
    
    game.state.units.push(unit);
    game.state.population++;
    
    this.emit('unit:training', { gameId, unitId, unitType, playerId });
    
    return unit;
  }

  /**
   * Handle player action - construct building
   */
  constructBuilding(gameId, playerId, buildingType, position) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    
    const cost = this.getBuildingCost(buildingType);
    if (!cost) {
      throw new Error('Invalid building type');
    }
    
    // Check if player has enough resources
    for (const [resource, amount] of Object.entries(cost)) {
      if (game.state.resources[resource] < amount) {
        throw new Error(`Not enough ${resource}`);
      }
    }
    
    // Deduct resources
    for (const [resource, amount] of Object.entries(cost)) {
      game.state.resources[resource] -= amount;
    }
    
    // Create building
    const buildingId = `${buildingType}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const building = {
      id: buildingId,
      type: buildingType,
      position,
      health: this.getBuildingMaxHealth(buildingType),
      maxHealth: this.getBuildingMaxHealth(buildingType),
      ownerId: playerId,
      isBuilding: true,
      buildProgress: 0,
      buildTime: this.getBuildTime(buildingType),
    };
    
    game.state.buildings.push(building);
    
    this.emit('building:constructing', { gameId, buildingId, buildingType, playerId });
    
    return building;
  }

  /**
   * Helper methods for unit/building stats
   */
  getUnitMaxHealth(unitType) {
    const healthMap = {
      villager: 50,
      scout: 60,
      warrior: 100,
      archer: 70,
    };
    return healthMap[unitType] || 50;
  }

  getUnitSpeed(unitType) {
    const speedMap = {
      villager: 1.2,
      scout: 2.5,  // Fast for exploration
      warrior: 1.5,
      archer: 1.3,
    };
    return speedMap[unitType] || 1.0;
  }

  getUnitBuildTime(unitType) {
    const buildTimeMap = {
      villager: 20,
      scout: 15,   // Quick to build
      warrior: 30,
      archer: 25,
    };
    return buildTimeMap[unitType] || 30;
  }

  getBuildingCost(buildingType) {
    const costs = {
      town_center: { wood: 500, stone: 300 },
      barracks: { wood: 150, gold: 50 },
      farm: { wood: 60 },
      lumber_mill: { wood: 100, gold: 50 },
      mining_camp: { wood: 100, gold: 50 },
      house: { wood: 50 },
    };
    return costs[buildingType] || null;
  }

  getBuildingMaxHealth(buildingType) {
    const healthMap = {
      town_center: 1000,
      barracks: 500,
      farm: 200,
      lumber_mill: 300,
      mining_camp: 300,
      house: 250,
    };
    return healthMap[buildingType] || 300;
  }

  /**
   * Handle player action - move unit
   */
  moveUnit(gameId, playerId, unitId, targetPosition) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    
    const unit = game.state.units.find(u => u.id === unitId);
    if (!unit) {
      throw new Error('Unit not found');
    }
    
    if (unit.ownerId !== playerId) {
      throw new Error('Not your unit');
    }
    
    unit.targetPosition = targetPosition;
    unit.isMoving = true;
    
    this.emit('unit:moving', { gameId, unitId, targetPosition });
  }

  /**
   * Handle player action - gather resource
   */
  gatherResource(gameId, playerId, unitId, resourceType, resourcePosition) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    
    const unit = game.state.units.find(u => u.id === unitId);
    if (!unit) {
      throw new Error('Unit not found');
    }
    
    if (unit.ownerId !== playerId) {
      throw new Error('Not your unit');
    }
    
    unit.isGathering = true;
    unit.targetResource = resourceType;
    unit.resourcePosition = resourcePosition;
    
    this.emit('unit:gathering', { gameId, unitId, resourceType });
  }

  /**
   * Get game state
   */
  getGameState(gameId) {
    const game = this.games.get(gameId);
    if (!game) {
      throw new Error('Game not found');
    }
    return game.state;
  }

  /**
   * Stop game loop
   */
  stopGameLoop() {
    if (this.gameLoopInterval) {
      clearInterval(this.gameLoopInterval);
      this.gameLoopInterval = null;
    }
  }

  /**
   * Clean up ended games
   */
  cleanupEndedGames() {
    const now = Date.now();
    const maxAge = 1000 * 60 * 60; // 1 hour
    
    this.games.forEach((game, gameId) => {
      if (game.status === 'ended' && now - game.createdAt > maxAge) {
        this.games.delete(gameId);
        this.emit('game:cleaned', { gameId });
      }
    });
  }
}

module.exports = GameServer;
