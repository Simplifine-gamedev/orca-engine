const express = require('express');
const http = require('http');
const socketIO = require('socket.io');

class GameServer {
  constructor(port = 3001) {
    this.port = port;
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = socketIO(this.server, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST'],
      },
    });

    // Game state
    this.lairs = new Map();
    this.mobs = new Map();
    this.players = new Map();
    this.lairSpawnTimers = new Map();

    // Configuration
    this.tickRate = 1000; // 1 second
    this.gameLoopInterval = null;

    this.setupMiddleware();
    this.setupSocketHandlers();
  }

  setupMiddleware() {
    this.app.use(express.json());

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'ok',
        uptime: process.uptime(),
        lairs: this.lairs.size,
        mobs: this.mobs.size,
        players: this.players.size,
      });
    });

    // Get game state endpoint
    this.app.get('/state', (req, res) => {
      res.json({
        lairs: Array.from(this.lairs.values()),
        mobs: Array.from(this.mobs.values()),
        players: this.players.size,
      });
    });
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`Player connected: ${socket.id}`);

      // Initialize player
      this.players.set(socket.id, {
        id: socket.id,
        connectedAt: Date.now(),
      });

      // Send initial game state to the connected player
      socket.emit('game:state', {
        lairs: Array.from(this.lairs.values()),
        mobs: Array.from(this.mobs.values()),
      });

      // Lair events
      socket.on('lair:create', (data) => {
        const lair = this.createLair(data.type, data.position);
        this.io.emit('lair:created', lair);
      });

      socket.on('lair:damage', (data) => {
        const result = this.damageLair(data.lairId, data.damage);
        if (result) {
          this.io.emit('lair:damaged', {
            lairId: data.lairId,
            health: result.health,
            destroyed: result.destroyed,
            loot: result.loot,
          });
        }
      });

      socket.on('lair:destroy', (data) => {
        const loot = this.destroyLair(data.lairId);
        this.io.emit('lair:destroyed', {
          lairId: data.lairId,
          loot,
        });
      });

      // Mob events
      socket.on('mob:kill', (data) => {
        const success = this.killMob(data.mobId);
        if (success) {
          this.io.emit('mob:killed', { mobId: data.mobId });
        }
      });

      socket.on('mob:damage', (data) => {
        const mob = this.damageMob(data.mobId, data.damage);
        if (mob) {
          this.io.emit('mob:damaged', {
            mobId: data.mobId,
            health: mob.health,
            killed: mob.health <= 0,
          });
        }
      });

      // Admin commands
      socket.on('game:reset', () => {
        this.reset();
        this.io.emit('game:reset');
      });

      // Disconnect handler
      socket.on('disconnect', () => {
        console.log(`Player disconnected: ${socket.id}`);
        this.players.delete(socket.id);
      });
    });
  }

  createLair(type, position) {
    const lairId = `lair_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Lair configuration based on type
    const lairConfigs = {
      goblin_camp: {
        maxHealth: 500,
        spawnInterval: 30000,
        maxMobs: 5,
        spawnRadius: 100,
        mobTypes: ['goblin_warrior', 'goblin_archer'],
        lootTable: [
          { itemId: 'gold', chance: 1.0, quantity: { min: 50, max: 100 } },
          { itemId: 'goblin_dagger', chance: 0.5, quantity: { min: 1, max: 1 } },
        ],
      },
      ogre_cave: {
        maxHealth: 1200,
        spawnInterval: 60000,
        maxMobs: 3,
        spawnRadius: 150,
        mobTypes: ['ogre', 'cave_troll'],
        lootTable: [
          { itemId: 'gold', chance: 1.0, quantity: { min: 200, max: 500 } },
          { itemId: 'ogre_club', chance: 0.4, quantity: { min: 1, max: 1 } },
        ],
      },
      wolf_den: {
        maxHealth: 400,
        spawnInterval: 20000,
        maxMobs: 8,
        spawnRadius: 120,
        mobTypes: ['wolf', 'dire_wolf'],
        lootTable: [
          { itemId: 'wolf_pelt', chance: 0.8, quantity: { min: 1, max: 3 } },
        ],
      },
      bandit_hideout: {
        maxHealth: 600,
        spawnInterval: 40000,
        maxMobs: 6,
        spawnRadius: 100,
        mobTypes: ['bandit', 'bandit_rogue'],
        lootTable: [
          { itemId: 'gold', chance: 1.0, quantity: { min: 100, max: 300 } },
          { itemId: 'stolen_goods', chance: 0.7, quantity: { min: 1, max: 5 } },
        ],
      },
      undead_crypt: {
        maxHealth: 800,
        spawnInterval: 45000,
        maxMobs: 10,
        spawnRadius: 140,
        mobTypes: ['skeleton', 'zombie', 'ghoul'],
        lootTable: [
          { itemId: 'bone_fragments', chance: 1.0, quantity: { min: 5, max: 15 } },
          { itemId: 'cursed_amulet', chance: 0.3, quantity: { min: 1, max: 1 } },
        ],
      },
    };

    const config = lairConfigs[type] || lairConfigs.goblin_camp;

    const lair = {
      id: lairId,
      type,
      position,
      health: config.maxHealth,
      maxHealth: config.maxHealth,
      spawnInterval: config.spawnInterval,
      maxMobs: config.maxMobs,
      spawnRadius: config.spawnRadius,
      mobTypes: config.mobTypes,
      lootTable: config.lootTable,
      destroyed: false,
      createdAt: Date.now(),
    };

    this.lairs.set(lairId, lair);

    // Initialize spawn timer
    this.lairSpawnTimers.set(lairId, {
      lairId,
      nextSpawnTime: Date.now() + config.spawnInterval,
      currentMobCount: 0,
    });

    console.log(`Lair created: ${lairId} (${type})`);
    return lair;
  }

  destroyLair(lairId) {
    const lair = this.lairs.get(lairId);
    if (!lair || lair.destroyed) {
      return [];
    }

    // Roll for loot
    const loot = this.rollLoot(lair.lootTable);

    // Mark as destroyed
    lair.destroyed = true;
    lair.health = 0;

    // Kill all mobs from this lair
    const mobsToKill = Array.from(this.mobs.values())
      .filter((mob) => mob.lairId === lairId)
      .map((mob) => mob.id);

    mobsToKill.forEach((mobId) => {
      this.killMob(mobId);
    });

    // Remove spawn timer
    this.lairSpawnTimers.delete(lairId);

    console.log(`Lair destroyed: ${lairId}, dropped ${loot.length} items`);
    return loot;
  }

  damageLair(lairId, damage) {
    const lair = this.lairs.get(lairId);
    if (!lair || lair.destroyed) {
      return null;
    }

    lair.health = Math.max(0, lair.health - damage);
    const destroyed = lair.health <= 0;

    let loot = [];
    if (destroyed) {
      loot = this.destroyLair(lairId);
    }

    return {
      health: lair.health,
      destroyed,
      loot,
    };
  }

  spawnMob(lairId) {
    const lair = this.lairs.get(lairId);
    const timer = this.lairSpawnTimers.get(lairId);

    if (!lair || lair.destroyed || !timer) {
      return null;
    }

    // Check mob count
    const currentMobs = Array.from(this.mobs.values()).filter(
      (mob) => mob.lairId === lairId
    );

    if (currentMobs.length >= lair.maxMobs) {
      return null;
    }

    // Select random mob type
    const mobType = lair.mobTypes[Math.floor(Math.random() * lair.mobTypes.length)];

    // Generate mob ID
    const mobId = `mob_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Calculate position within spawn radius
    const angle = Math.random() * Math.PI * 2;
    const distance = Math.random() * lair.spawnRadius;
    const position = {
      x: lair.position.x + Math.cos(angle) * distance,
      y: lair.position.y + Math.sin(angle) * distance,
    };

    // Calculate mob stats
    const level = Math.floor(Math.random() * 5) + 1;
    const maxHealth = this.calculateMobHealth(mobType, level);

    const mob = {
      id: mobId,
      type: mobType,
      level,
      position,
      health: maxHealth,
      maxHealth,
      lairId,
      spawnTime: Date.now(),
    };

    this.mobs.set(mobId, mob);

    // Update timer
    timer.currentMobCount = currentMobs.length + 1;
    timer.nextSpawnTime = Date.now() + lair.spawnInterval;

    // Broadcast to all clients
    this.io.emit('mob:spawned', mob);

    console.log(`Mob spawned: ${mobId} (${mobType}) from lair ${lairId}`);
    return mob;
  }

  killMob(mobId) {
    const mob = this.mobs.get(mobId);
    if (!mob) {
      return false;
    }

    // Update lair spawn timer
    if (mob.lairId) {
      const timer = this.lairSpawnTimers.get(mob.lairId);
      if (timer) {
        timer.currentMobCount = Math.max(0, timer.currentMobCount - 1);
      }
    }

    this.mobs.delete(mobId);
    console.log(`Mob killed: ${mobId}`);
    return true;
  }

  damageMob(mobId, damage) {
    const mob = this.mobs.get(mobId);
    if (!mob) {
      return null;
    }

    mob.health = Math.max(0, mob.health - damage);

    if (mob.health <= 0) {
      this.killMob(mobId);
    }

    return mob;
  }

  calculateMobHealth(mobType, level) {
    const baseHealth = {
      goblin_warrior: 50,
      goblin_archer: 40,
      ogre: 200,
      cave_troll: 150,
      wolf: 60,
      dire_wolf: 100,
      bandit: 70,
      bandit_rogue: 80,
      skeleton: 50,
      zombie: 80,
      ghoul: 110,
    };

    return (baseHealth[mobType] || 50) * level;
  }

  rollLoot(lootTable) {
    const loot = [];

    for (const entry of lootTable) {
      if (Math.random() < entry.chance) {
        const quantity =
          Math.floor(Math.random() * (entry.quantity.max - entry.quantity.min + 1)) +
          entry.quantity.min;

        loot.push({
          itemId: entry.itemId,
          quantity,
        });
      }
    }

    return loot;
  }

  startGameLoop() {
    if (this.gameLoopInterval) {
      return;
    }

    console.log('Game loop started');

    this.gameLoopInterval = setInterval(() => {
      this.tick();
    }, this.tickRate);
  }

  stopGameLoop() {
    if (this.gameLoopInterval) {
      clearInterval(this.gameLoopInterval);
      this.gameLoopInterval = null;
      console.log('Game loop stopped');
    }
  }

  tick() {
    const now = Date.now();

    // Process lair spawning
    for (const [lairId, timer] of this.lairSpawnTimers.entries()) {
      const lair = this.lairs.get(lairId);

      if (lair && !lair.destroyed && now >= timer.nextSpawnTime) {
        this.spawnMob(lairId);
      }
    }
  }

  reset() {
    this.stopGameLoop();
    this.lairs.clear();
    this.mobs.clear();
    this.lairSpawnTimers.clear();
    console.log('Game state reset');
  }

  start() {
    this.server.listen(this.port, () => {
      console.log(`Game server listening on port ${this.port}`);
      this.startGameLoop();
    });
  }

  stop() {
    this.stopGameLoop();
    this.server.close();
  }
}

// Export for use as module
module.exports = GameServer;

// Run server if executed directly
if (require.main === module) {
  const port = process.env.PORT || 3001;
  const server = new GameServer(port);
  server.start();

  // Graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nShutting down gracefully...');
    server.stop();
    process.exit(0);
  });
}
