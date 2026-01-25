const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});

app.use(cors());
app.use(express.json());

// Game state
const gameState = {
  lairs: new Map(),
  mobs: new Map(),
  players: new Map(),
};

// Lair configurations
const LAIR_CONFIGS = {
  goblin_camp: {
    maxHealth: 500,
    spawnInterval: 30000,
    mobType: 'goblin',
    maxMobs: 8,
    lootTable: [
      { item: 'gold', quantity: 50, dropChance: 1.0 },
      { item: 'goblin_dagger', quantity: 1, dropChance: 0.3 },
      { item: 'leather_scraps', quantity: 5, dropChance: 0.7 },
    ],
  },
  ogre_cave: {
    maxHealth: 1200,
    spawnInterval: 60000,
    mobType: 'ogre',
    maxMobs: 4,
    lootTable: [
      { item: 'gold', quantity: 150, dropChance: 1.0 },
      { item: 'ogre_club', quantity: 1, dropChance: 0.4 },
      { item: 'thick_hide', quantity: 3, dropChance: 0.8 },
    ],
  },
  undead_crypt: {
    maxHealth: 800,
    spawnInterval: 25000,
    mobType: 'skeleton',
    maxMobs: 12,
    lootTable: [
      { item: 'gold', quantity: 80, dropChance: 1.0 },
      { item: 'bone_sword', quantity: 1, dropChance: 0.25 },
      { item: 'soul_essence', quantity: 2, dropChance: 0.6 },
    ],
  },
  dragon_nest: {
    maxHealth: 3000,
    spawnInterval: 120000,
    mobType: 'dragon',
    maxMobs: 2,
    lootTable: [
      { item: 'gold', quantity: 500, dropChance: 1.0 },
      { item: 'dragon_scale', quantity: 5, dropChance: 0.9 },
      { item: 'legendary_weapon', quantity: 1, dropChance: 0.1 },
    ],
  },
};

// Spawn timers
const spawnTimers = new Map();

// Helper functions
function generateId() {
  return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getMobsByLair(lairId) {
  return Array.from(gameState.mobs.values()).filter(
    mob => mob.lairId === lairId && mob.isAlive
  );
}

function spawnMob(lair) {
  const currentMobs = getMobsByLair(lair.id);
  
  if (currentMobs.length >= lair.maxMobs) {
    return null;
  }

  const mob = {
    id: generateId(),
    type: lair.mobType,
    lairId: lair.id,
    position: {
      x: lair.position.x + (Math.random() - 0.5) * 100,
      y: lair.position.y + (Math.random() - 0.5) * 100,
    },
    health: 100,
    maxHealth: 100,
    isAlive: true,
    spawnTime: Date.now(),
  };

  gameState.mobs.set(mob.id, mob);
  
  // Broadcast mob spawn to all clients
  io.emit('mob:spawned', mob);
  
  console.log(`Spawned ${mob.type} from lair ${lair.id}`);
  
  return mob;
}

function startLairSpawning(lairId) {
  const lair = gameState.lairs.get(lairId);
  if (!lair || lair.isDestroyed) return;

  // Clear existing timer if any
  if (spawnTimers.has(lairId)) {
    clearInterval(spawnTimers.get(lairId));
  }

  // Start new spawn timer
  const timer = setInterval(() => {
    const currentLair = gameState.lairs.get(lairId);
    if (!currentLair || currentLair.isDestroyed) {
      clearInterval(timer);
      spawnTimers.delete(lairId);
      return;
    }

    spawnMob(currentLair);
  }, lair.spawnInterval);

  spawnTimers.set(lairId, timer);
  
  // Initial spawn
  spawnMob(lair);
}

function stopLairSpawning(lairId) {
  if (spawnTimers.has(lairId)) {
    clearInterval(spawnTimers.get(lairId));
    spawnTimers.delete(lairId);
  }
}

function generateLoot(lootTable) {
  return lootTable
    .filter(loot => Math.random() < loot.dropChance)
    .map(loot => ({
      ...loot,
      quantity: Math.floor(loot.quantity * (0.8 + Math.random() * 0.4)),
    }));
}

// REST API endpoints
app.get('/api/game-state', (req, res) => {
  res.json({
    lairs: Array.from(gameState.lairs.values()),
    mobs: Array.from(gameState.mobs.values()),
    players: Array.from(gameState.players.values()),
  });
});

app.post('/api/lairs', (req, res) => {
  const { type, position } = req.body;
  
  if (!LAIR_CONFIGS[type]) {
    return res.status(400).json({ error: 'Invalid lair type' });
  }

  const config = LAIR_CONFIGS[type];
  const lair = {
    id: generateId(),
    type,
    position,
    health: config.maxHealth,
    maxHealth: config.maxHealth,
    spawnInterval: config.spawnInterval,
    mobType: config.mobType,
    maxMobs: config.maxMobs,
    lootTable: config.lootTable,
    isDestroyed: false,
    createdAt: Date.now(),
  };

  gameState.lairs.set(lair.id, lair);
  startLairSpawning(lair.id);
  
  io.emit('lair:created', lair);
  
  res.json(lair);
});

app.delete('/api/lairs/:lairId', (req, res) => {
  const { lairId } = req.params;
  const lair = gameState.lairs.get(lairId);
  
  if (!lair) {
    return res.status(404).json({ error: 'Lair not found' });
  }

  stopLairSpawning(lairId);
  gameState.lairs.delete(lairId);
  
  // Remove all mobs from this lair
  Array.from(gameState.mobs.values())
    .filter(mob => mob.lairId === lairId)
    .forEach(mob => gameState.mobs.delete(mob.id));
  
  io.emit('lair:removed', { lairId });
  
  res.json({ success: true });
});

// Socket.IO event handlers
io.on('connection', (socket) => {
  console.log(`Player connected: ${socket.id}`);

  // Add player to game state
  gameState.players.set(socket.id, {
    id: socket.id,
    connectedAt: Date.now(),
  });

  // Send current game state to newly connected player
  socket.emit('game:state', {
    lairs: Array.from(gameState.lairs.values()),
    mobs: Array.from(gameState.mobs.values()),
  });

  // Handle lair damage
  socket.on('lair:damage', ({ lairId, damage }) => {
    const lair = gameState.lairs.get(lairId);
    if (!lair || lair.isDestroyed) return;

    lair.health = Math.max(0, lair.health - damage);

    if (lair.health <= 0) {
      lair.isDestroyed = true;
      stopLairSpawning(lairId);

      const loot = generateLoot(lair.lootTable);
      
      io.emit('lair:destroyed', {
        lairId,
        position: lair.position,
        loot,
      });

      console.log(`Lair ${lairId} destroyed, dropped loot:`, loot);
    } else {
      io.emit('lair:damaged', {
        lairId,
        health: lair.health,
        maxHealth: lair.maxHealth,
      });
    }
  });

  // Handle mob damage
  socket.on('mob:damage', ({ mobId, damage }) => {
    const mob = gameState.mobs.get(mobId);
    if (!mob || !mob.isAlive) return;

    mob.health = Math.max(0, mob.health - damage);

    if (mob.health <= 0) {
      mob.isAlive = false;
      
      io.emit('mob:killed', {
        mobId,
        killerId: socket.id,
      });

      // Remove mob after delay
      setTimeout(() => {
        gameState.mobs.delete(mobId);
      }, 5000);

      console.log(`Mob ${mobId} killed by ${socket.id}`);
    } else {
      io.emit('mob:damaged', {
        mobId,
        health: mob.health,
        maxHealth: mob.maxHealth,
      });
    }
  });

  // Handle mob movement
  socket.on('mob:move', ({ mobId, position }) => {
    const mob = gameState.mobs.get(mobId);
    if (!mob || !mob.isAlive) return;

    mob.position = position;
    socket.broadcast.emit('mob:moved', { mobId, position });
  });

  // Handle disconnect
  socket.on('disconnect', () => {
    console.log(`Player disconnected: ${socket.id}`);
    gameState.players.delete(socket.id);
  });
});

// Game loop for mob AI (simple patrol/idle behavior)
setInterval(() => {
  Array.from(gameState.mobs.values())
    .filter(mob => mob.isAlive)
    .forEach(mob => {
      // Simple random movement
      if (Math.random() < 0.3) {
        const lair = gameState.lairs.get(mob.lairId);
        if (lair) {
          const maxDistance = 150;
          mob.position.x = lair.position.x + (Math.random() - 0.5) * maxDistance;
          mob.position.y = lair.position.y + (Math.random() - 0.5) * maxDistance;
          
          io.emit('mob:moved', {
            mobId: mob.id,
            position: mob.position,
          });
        }
      }
    });
}, 5000);

// Cleanup destroyed lairs periodically
setInterval(() => {
  const now = Date.now();
  const LAIR_CLEANUP_TIME = 60000; // 1 minute

  Array.from(gameState.lairs.values())
    .filter(lair => lair.isDestroyed && (now - lair.destroyedAt) > LAIR_CLEANUP_TIME)
    .forEach(lair => {
      gameState.lairs.delete(lair.id);
      console.log(`Cleaned up destroyed lair ${lair.id}`);
    });
}, 30000);

// Start server
const PORT = process.env.PORT || 3001;

server.listen(PORT, () => {
  console.log(`Game server running on port ${PORT}`);
  console.log(`Available lair types: ${Object.keys(LAIR_CONFIGS).join(', ')}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  
  // Clear all spawn timers
  spawnTimers.forEach(timer => clearInterval(timer));
  
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

module.exports = { app, server, io };
