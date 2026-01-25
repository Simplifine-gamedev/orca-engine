// Game Server - Handles server-side game logic and synchronization
// ORC-168: Early game pacing improvements

const EventEmitter = require('events');

// Import game constants (in a real setup, these would be shared)
const STARTING_RESOURCES = {
  food: 400,    // Increased from 200
  wood: 300,    // Increased from 150
  stone: 200,   // Increased from 100
  gold: 100,    // Increased from 50
};

const RESOURCE_GATHERING_RATES = {
  food: 15,     // Increased from 10 (50% faster)
  wood: 12,     // Increased from 8 (50% faster)
  stone: 9,     // Increased from 6 (50% faster)
  gold: 6,      // Increased from 4 (50% faster)
};

const UNIT_TYPES = {
  scout: {
    cost: { food: 50, wood: 0, stone: 0, gold: 0 },
    buildTime: 5,
    speed: 8,
    health: 75,
    attackDamage: 5,
    visionRange: 12,
  },
  worker: {
    cost: { food: 50, wood: 0, stone: 0, gold: 0 },
    buildTime: 10,  // Reduced from 15
    speed: 4,
    health: 100,
    attackDamage: 5,
    visionRange: 6,
  },
  soldier: {
    cost: { food: 60, wood: 20, stone: 0, gold: 0 },
    buildTime: 12,  // Reduced from 20
    speed: 5,
    health: 150,
    attackDamage: 15,
    visionRange: 8,
  },
  archer: {
    cost: { food: 50, wood: 30, stone: 0, gold: 10 },
    buildTime: 15,  // Reduced from 25
    speed: 4,
    health: 100,
    attackDamage: 20,
    visionRange: 10,
  },
};

const BUILDING_TYPES = {
  townCenter: {
    cost: { food: 0, wood: 0, stone: 0, gold: 0 },
    buildTime: 0,
    health: 2000,
    produces: ["worker", "scout"],
  },
  house: {
    cost: { food: 0, wood: 50, stone: 0, gold: 0 },
    buildTime: 15,  // Reduced from 30
    health: 500,
    populationBonus: 5,
  },
  barracks: {
    cost: { food: 0, wood: 100, stone: 50, gold: 0 },
    buildTime: 25,  // Reduced from 45
    health: 1000,
    produces: ["soldier", "archer"],
  },
  farm: {
    cost: { food: 0, wood: 75, stone: 0, gold: 0 },
    buildTime: 20,  // Reduced from 35
    health: 500,
    resourceGeneration: { food: 8, wood: 0, stone: 0, gold: 0 },
  },
  lumberMill: {
    cost: { food: 0, wood: 100, stone: 0, gold: 0 },
    buildTime: 25,  // Reduced from 40
    health: 750,
    gatheringBonus: { food: 0, wood: 1.25, stone: 0, gold: 0 },
  },
  stoneMine: {
    cost: { food: 0, wood: 75, stone: 0, gold: 0 },
    buildTime: 30,  // Reduced from 50
    health: 1000,
    resourceGeneration: { food: 0, wood: 0, stone: 5, gold: 0 },
  },
};

// Early game quests for better pacing
const EARLY_GAME_QUESTS = [
  {
    id: "tutorial_1",
    title: "First Steps",
    description: "Train your first scout to explore the map",
    objectives: [{ type: "train_unit", target: "scout", required: 1 }],
    rewards: { food: 100, wood: 50, stone: 0, gold: 0 },
  },
  {
    id: "tutorial_2",
    title: "Gather Resources",
    description: "Collect 500 food to sustain your civilization",
    objectives: [{ type: "collect_resource", target: "food", required: 500 }],
    rewards: { food: 0, wood: 100, stone: 50, gold: 25 },
  },
  {
    id: "tutorial_3",
    title: "Build Your Base",
    description: "Construct your first house to increase population capacity",
    objectives: [{ type: "build_building", target: "house", required: 1 }],
    rewards: { food: 150, wood: 100, stone: 50, gold: 0 },
  },
  {
    id: "tutorial_4",
    title: "Expand Economy",
    description: "Build a farm to generate food automatically",
    objectives: [{ type: "build_building", target: "farm", required: 1 }],
    rewards: { food: 200, wood: 150, stone: 100, gold: 50 },
  },
  {
    id: "tutorial_5",
    title: "Train Your Army",
    description: "Build a barracks and train 2 soldiers",
    objectives: [
      { type: "build_building", target: "barracks", required: 1 },
      { type: "train_unit", target: "soldier", required: 2 },
    ],
    rewards: { food: 300, wood: 200, stone: 150, gold: 100 },
  },
];

class GameServer extends EventEmitter {
  constructor() {
    super();
    this.games = new Map();
    this.tickRate = 50; // 50ms = 20 ticks per second
    this.isRunning = false;
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.gameLoop();
    console.log('GameServer started with improved early game pacing (ORC-168)');
  }

  stop() {
    this.isRunning = false;
    console.log('GameServer stopped');
  }

  createGame(gameId, playerId) {
    const game = {
      id: gameId,
      players: [playerId],
      state: {
        resources: { ...STARTING_RESOURCES },
        units: [],
        buildings: [],
        gameTime: 0,
      },
      quests: EARLY_GAME_QUESTS.map((q) => ({
        ...q,
        objectives: q.objectives.map((obj) => ({ ...obj, current: 0 })),
        completed: false,
      })),
      lastUpdate: Date.now(),
    };

    // Initialize starting units and buildings
    this.initializeGame(game);
    this.games.set(gameId, game);

    console.log(`Game ${gameId} created for player ${playerId}`);
    return game;
  }

  initializeGame(game) {
    // Start with town center
    game.state.buildings.push({
      id: `tc_1`,
      type: 'townCenter',
      position: { x: 50, y: 50 },
      buildTime: 0,
      health: 2000,
    });

    // Start with 3 workers for faster early game (was 1 worker)
    for (let i = 0; i < 3; i++) {
      game.state.units.push({
        id: `worker_${i}`,
        type: 'worker',
        position: { x: 50 + i * 2, y: 50 + i * 2 },
        health: 100,
        speed: 4,
        state: 'idle',
      });
    }

    console.log(`Game initialized with ${game.state.units.length} starting workers`);
  }

  getGame(gameId) {
    return this.games.get(gameId);
  }

  handlePlayerAction(gameId, playerId, action) {
    const game = this.games.get(gameId);
    if (!game) {
      return { success: false, error: 'Game not found' };
    }

    if (!game.players.includes(playerId)) {
      return { success: false, error: 'Player not in game' };
    }

    switch (action.type) {
      case 'train_unit':
        return this.handleTrainUnit(game, action.unitType);
      case 'build_building':
        return this.handleBuildBuilding(game, action.buildingType, action.position);
      case 'gather_resource':
        return this.handleGatherResource(game, action.unitId, action.resourceType);
      case 'move_unit':
        return this.handleMoveUnit(game, action.unitId, action.position);
      default:
        return { success: false, error: 'Unknown action type' };
    }
  }

  handleTrainUnit(game, unitType) {
    const unitConfig = UNIT_TYPES[unitType];
    if (!unitConfig) {
      return { success: false, error: 'Invalid unit type' };
    }

    // Check if player has enough resources
    if (!this.hasEnoughResources(game.state.resources, unitConfig.cost)) {
      return { success: false, error: 'Not enough resources' };
    }

    // Deduct resources
    this.spendResources(game.state.resources, unitConfig.cost);

    // Create unit (in real game, this would be queued with build time)
    const unit = {
      id: `${unitType}_${Date.now()}`,
      type: unitType,
      position: { x: 50, y: 50 },
      health: unitConfig.health,
      speed: unitConfig.speed,
      state: 'idle',
    };

    game.state.units.push(unit);

    // Update quest progress
    this.updateQuestProgress(game, 'train_unit', unitType);

    this.emit('unitTrained', { gameId: game.id, unit });
    return { success: true, unit };
  }

  handleBuildBuilding(game, buildingType, position) {
    const buildingConfig = BUILDING_TYPES[buildingType];
    if (!buildingConfig) {
      return { success: false, error: 'Invalid building type' };
    }

    // Check if player has enough resources
    if (!this.hasEnoughResources(game.state.resources, buildingConfig.cost)) {
      return { success: false, error: 'Not enough resources' };
    }

    // Deduct resources
    this.spendResources(game.state.resources, buildingConfig.cost);

    // Create building (in real game, this would be constructed over time)
    const building = {
      id: `${buildingType}_${Date.now()}`,
      type: buildingType,
      position: position || { x: 50, y: 50 },
      buildTime: buildingConfig.buildTime,
      health: buildingConfig.health,
    };

    game.state.buildings.push(building);

    // Update quest progress
    this.updateQuestProgress(game, 'build_building', buildingType);

    this.emit('buildingConstructed', { gameId: game.id, building });
    return { success: true, building };
  }

  handleGatherResource(game, unitId, resourceType) {
    const unit = game.state.units.find((u) => u.id === unitId);
    if (!unit) {
      return { success: false, error: 'Unit not found' };
    }

    if (unit.type !== 'worker') {
      return { success: false, error: 'Only workers can gather resources' };
    }

    // Apply gathering rate (improved by 50%)
    const gatherAmount = RESOURCE_GATHERING_RATES[resourceType] || 0;
    game.state.resources[resourceType] += gatherAmount;

    unit.state = 'gathering';
    unit.gatheringResource = resourceType;

    // Update quest progress for resource collection
    this.updateQuestProgress(game, 'collect_resource', resourceType, game.state.resources[resourceType]);

    this.emit('resourceGathered', { 
      gameId: game.id, 
      unitId, 
      resourceType, 
      amount: gatherAmount 
    });

    return { success: true, amount: gatherAmount };
  }

  handleMoveUnit(game, unitId, position) {
    const unit = game.state.units.find((u) => u.id === unitId);
    if (!unit) {
      return { success: false, error: 'Unit not found' };
    }

    unit.position = position;
    unit.state = 'moving';

    return { success: true };
  }

  hasEnoughResources(currentResources, cost) {
    return Object.keys(cost).every(
      (resource) => currentResources[resource] >= cost[resource]
    );
  }

  spendResources(currentResources, cost) {
    Object.keys(cost).forEach((resource) => {
      currentResources[resource] -= cost[resource];
    });
  }

  updateQuestProgress(game, actionType, target, value) {
    game.quests.forEach((quest) => {
      if (quest.completed) return;

      quest.objectives.forEach((objective, index) => {
        if (objective.type === actionType && objective.target === target) {
          if (value !== undefined) {
            objective.current = value;
          } else {
            objective.current += 1;
          }

          // Check if objective is complete
          if (objective.current >= objective.required) {
            console.log(`Quest objective completed: ${quest.title} - ${objective.type}`);
          }
        }
      });

      // Check if all objectives are complete
      const allComplete = quest.objectives.every(
        (obj) => obj.current >= obj.required
      );

      if (allComplete && !quest.completed) {
        quest.completed = true;
        // Award quest rewards
        Object.keys(quest.rewards).forEach((resource) => {
          game.state.resources[resource] += quest.rewards[resource];
        });

        console.log(`Quest completed: ${quest.title}`);
        this.emit('questCompleted', { gameId: game.id, quest });
      }
    });
  }

  gameLoop() {
    if (!this.isRunning) return;

    const now = Date.now();

    this.games.forEach((game) => {
      const deltaTime = (now - game.lastUpdate) / 1000; // Convert to seconds
      game.lastUpdate = now;

      // Update game time
      game.state.gameTime += deltaTime;

      // Auto-generate resources from buildings
      game.state.buildings.forEach((building) => {
        const buildingConfig = BUILDING_TYPES[building.type];
        if (buildingConfig && buildingConfig.resourceGeneration) {
          Object.keys(buildingConfig.resourceGeneration).forEach((resource) => {
            game.state.resources[resource] +=
              buildingConfig.resourceGeneration[resource] * deltaTime;
          });
        }
      });

      // Emit game state update
      this.emit('gameUpdate', { gameId: game.id, state: game.state });
    });

    setTimeout(() => this.gameLoop(), this.tickRate);
  }

  getGameState(gameId) {
    const game = this.games.get(gameId);
    return game ? game.state : null;
  }

  getActiveQuests(gameId) {
    const game = this.games.get(gameId);
    return game ? game.quests.filter((q) => !q.completed) : [];
  }

  getCompletedQuests(gameId) {
    const game = this.games.get(gameId);
    return game ? game.quests.filter((q) => q.completed) : [];
  }
}

module.exports = GameServer;

// Example usage:
if (require.main === module) {
  const server = new GameServer();
  server.start();

  // Create a test game
  const gameId = 'test_game_1';
  const playerId = 'player_1';
  const game = server.createGame(gameId, playerId);

  console.log('\n=== Initial Game State ===');
  console.log('Starting Resources:', game.state.resources);
  console.log('Starting Units:', game.state.units.length);
  console.log('Active Quests:', server.getActiveQuests(gameId).length);

  // Simulate some actions
  setTimeout(() => {
    console.log('\n=== Testing Unit Training ===');
    const result = server.handlePlayerAction(gameId, playerId, {
      type: 'train_unit',
      unitType: 'scout',
    });
    console.log('Train scout result:', result);
    console.log('Resources after training:', game.state.resources);
    console.log('Total units:', game.state.units.length);
  }, 1000);

  setTimeout(() => {
    console.log('\n=== Testing Building Construction ===');
    const result = server.handlePlayerAction(gameId, playerId, {
      type: 'build_building',
      buildingType: 'house',
      position: { x: 60, y: 60 },
    });
    console.log('Build house result:', result);
    console.log('Resources after building:', game.state.resources);
    console.log('Total buildings:', game.state.buildings.length);
  }, 2000);

  // Listen for events
  server.on('unitTrained', (data) => {
    console.log(`\n[EVENT] Unit trained: ${data.unit.type} in game ${data.gameId}`);
  });

  server.on('buildingConstructed', (data) => {
    console.log(`[EVENT] Building constructed: ${data.building.type} in game ${data.gameId}`);
  });

  server.on('questCompleted', (data) => {
    console.log(`[EVENT] Quest completed: ${data.quest.title} in game ${data.gameId}`);
    console.log(`[EVENT] Rewards: ${JSON.stringify(data.quest.rewards)}`);
  });

  // Stop server after demo
  setTimeout(() => {
    console.log('\n=== Final Game State ===');
    console.log('Current Resources:', game.state.resources);
    console.log('Active Quests:', server.getActiveQuests(gameId).map(q => q.title));
    console.log('Completed Quests:', server.getCompletedQuests(gameId).map(q => q.title));
    server.stop();
    process.exit(0);
  }, 5000);
}
