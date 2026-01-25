/**
 * Game Server - Server-authoritative combat and mob management
 * Handles all combat calculations to prevent cheating
 */

const { MOB_CONFIGS, calculateDamage, createMobInstance } = require('../src/store/mobStore.ts');

/**
 * Game Server Class
 * Manages game state, combat, and mob spawning
 */
class GameServer {
  constructor() {
    this.mobs = new Map(); // mobId -> mob instance
    this.players = new Map(); // playerId -> player data
    this.combatLog = [];
    this.nextMobId = 1;
    this.tickRate = 60; // Game updates per second
    this.lastTick = Date.now();
  }

  /**
   * Initialize the game server
   */
  initialize() {
    console.log('[GameServer] Initializing...');
    console.log('[GameServer] Loaded mob types:', Object.keys(MOB_CONFIGS).length);
    
    // Start game loop
    this.startGameLoop();
  }

  /**
   * Start the main game loop
   */
  startGameLoop() {
    setInterval(() => {
      this.tick();
    }, 1000 / this.tickRate);
  }

  /**
   * Main game tick - processes all game logic
   */
  tick() {
    const now = Date.now();
    const deltaTime = (now - this.lastTick) / 1000; // Convert to seconds
    this.lastTick = now;

    // Update mobs
    this.updateMobs(deltaTime);

    // Process combat
    this.processCombat(deltaTime);

    // Clean up dead mobs
    this.cleanupDeadMobs();
  }

  /**
   * Spawn a new mob at specified position
   * @param {string} mobType - Type of mob to spawn
   * @param {Object} position - {x, y} position
   * @returns {string} Mob ID
   */
  spawnMob(mobType, position) {
    const config = MOB_CONFIGS[mobType];
    if (!config) {
      console.error(`[GameServer] Invalid mob type: ${mobType}`);
      return null;
    }

    const mobId = `mob_${this.nextMobId++}`;
    const mobStats = createMobInstance(mobType);
    
    if (!mobStats) {
      console.error(`[GameServer] Failed to create mob instance: ${mobType}`);
      return null;
    }

    const mob = {
      id: mobId,
      type: mobType,
      config: config,
      stats: mobStats,
      position: position,
      target: null,
      lastAttackTime: 0,
      isDead: false,
      spawnTime: Date.now()
    };

    this.mobs.set(mobId, mob);
    console.log(`[GameServer] Spawned ${config.name} (${mobId}) at`, position);
    
    return mobId;
  }

  /**
   * Apply damage to a mob with armor calculations
   * @param {string} mobId - Target mob ID
   * @param {number} baseDamage - Raw damage before armor
   * @param {string} attackerId - ID of attacker
   * @returns {Object} Combat result
   */
  applyDamageToMob(mobId, baseDamage, attackerId) {
    const mob = this.mobs.get(mobId);
    
    if (!mob || mob.isDead) {
      return { success: false, reason: 'Mob not found or already dead' };
    }

    // Calculate actual damage after armor mitigation
    const actualDamage = calculateDamage(
      baseDamage,
      mob.stats.armor,
      mob.stats.armorPercent
    );

    // Apply damage
    const oldHealth = mob.stats.health;
    mob.stats.health = Math.max(0, mob.stats.health - actualDamage);
    
    const result = {
      success: true,
      mobId: mobId,
      mobType: mob.type,
      baseDamage: baseDamage,
      actualDamage: actualDamage,
      damageReduced: baseDamage - actualDamage,
      oldHealth: oldHealth,
      newHealth: mob.stats.health,
      maxHealth: mob.stats.maxHealth,
      isDead: mob.stats.health <= 0,
      attackerId: attackerId
    };

    // Log the combat event
    this.logCombat(result);

    // Check if mob died
    if (mob.stats.health <= 0) {
      mob.isDead = true;
      this.handleMobDeath(mob, attackerId);
    }

    return result;
  }

  /**
   * Handle mob death - grant rewards, etc.
   * @param {Object} mob - The mob that died
   * @param {string} killerId - ID of the killer
   */
  handleMobDeath(mob, killerId) {
    console.log(`[GameServer] ${mob.config.name} (${mob.id}) was killed by ${killerId}`);
    
    // Grant rewards to killer
    const player = this.players.get(killerId);
    if (player) {
      player.xp += mob.stats.xpReward;
      player.gold += mob.stats.goldReward;
      player.kills++;
      
      console.log(`[GameServer] ${killerId} gained ${mob.stats.xpReward} XP and ${mob.stats.goldReward} gold`);
    }

    // Broadcast mob death event
    this.broadcastEvent('mob_death', {
      mobId: mob.id,
      mobType: mob.type,
      killerId: killerId,
      position: mob.position,
      rewards: {
        xp: mob.stats.xpReward,
        gold: mob.stats.goldReward
      }
    });
  }

  /**
   * Update all mobs (AI, movement, etc.)
   * @param {number} deltaTime - Time since last update in seconds
   */
  updateMobs(deltaTime) {
    for (const [mobId, mob] of this.mobs) {
      if (mob.isDead) continue;

      // Simple AI: Find nearest player target
      if (!mob.target || !this.players.has(mob.target)) {
        mob.target = this.findNearestPlayer(mob.position);
      }

      // Move towards target
      if (mob.target) {
        const targetPlayer = this.players.get(mob.target);
        if (targetPlayer) {
          this.moveMobTowardsTarget(mob, targetPlayer.position, deltaTime);
        }
      }
    }
  }

  /**
   * Move mob towards a target position
   * @param {Object} mob - The mob to move
   * @param {Object} targetPos - Target {x, y} position
   * @param {number} deltaTime - Time delta
   */
  moveMobTowardsTarget(mob, targetPos, deltaTime) {
    const dx = targetPos.x - mob.position.x;
    const dy = targetPos.y - mob.position.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance > mob.stats.attackRange) {
      // Move towards target
      const moveDistance = mob.stats.moveSpeed * deltaTime;
      const ratio = moveDistance / distance;
      
      mob.position.x += dx * ratio;
      mob.position.y += dy * ratio;
    }
  }

  /**
   * Process combat between mobs and players
   * @param {number} deltaTime - Time delta
   */
  processCombat(deltaTime) {
    const now = Date.now();

    for (const [mobId, mob] of this.mobs) {
      if (mob.isDead || !mob.target) continue;

      const targetPlayer = this.players.get(mob.target);
      if (!targetPlayer) continue;

      // Check if in attack range
      const dx = targetPlayer.position.x - mob.position.x;
      const dy = targetPlayer.position.y - mob.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance <= mob.stats.attackRange) {
        // Check attack cooldown
        const attackCooldown = 1000 / mob.stats.attackSpeed;
        if (now - mob.lastAttackTime >= attackCooldown) {
          this.mobAttackPlayer(mob, targetPlayer);
          mob.lastAttackTime = now;
        }
      }
    }
  }

  /**
   * Mob attacks a player
   * @param {Object} mob - Attacking mob
   * @param {Object} player - Target player
   */
  mobAttackPlayer(mob, player) {
    // Calculate damage to player (players can also have armor)
    const actualDamage = calculateDamage(
      mob.stats.damage,
      player.armor || 0,
      player.armorPercent || 0
    );

    player.health = Math.max(0, player.health - actualDamage);

    console.log(`[GameServer] ${mob.config.name} hit ${player.id} for ${actualDamage} damage`);

    // Broadcast attack event
    this.broadcastEvent('mob_attack', {
      mobId: mob.id,
      targetId: player.id,
      damage: actualDamage
    });

    // Check if player died
    if (player.health <= 0) {
      this.handlePlayerDeath(player);
    }
  }

  /**
   * Find nearest player to a position
   * @param {Object} position - {x, y} position
   * @returns {string|null} Player ID
   */
  findNearestPlayer(position) {
    let nearestPlayer = null;
    let nearestDistance = Infinity;

    for (const [playerId, player] of this.players) {
      const dx = player.position.x - position.x;
      const dy = player.position.y - position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestPlayer = playerId;
      }
    }

    return nearestPlayer;
  }

  /**
   * Clean up dead mobs from the game
   */
  cleanupDeadMobs() {
    const deadMobs = [];
    
    for (const [mobId, mob] of this.mobs) {
      if (mob.isDead) {
        // Wait a bit before removing (for animation, etc.)
        const timeSinceDeath = Date.now() - (mob.deathTime || Date.now());
        if (timeSinceDeath > 5000) { // 5 seconds
          deadMobs.push(mobId);
        }
      }
    }

    for (const mobId of deadMobs) {
      this.mobs.delete(mobId);
    }
  }

  /**
   * Handle player death
   * @param {Object} player - The player that died
   */
  handlePlayerDeath(player) {
    console.log(`[GameServer] Player ${player.id} died`);
    
    this.broadcastEvent('player_death', {
      playerId: player.id,
      position: player.position
    });
  }

  /**
   * Log combat event
   * @param {Object} event - Combat event data
   */
  logCombat(event) {
    this.combatLog.push({
      timestamp: Date.now(),
      ...event
    });

    // Keep only last 1000 events
    if (this.combatLog.length > 1000) {
      this.combatLog.shift();
    }
  }

  /**
   * Broadcast event to all clients (placeholder)
   * @param {string} eventType - Type of event
   * @param {Object} data - Event data
   */
  broadcastEvent(eventType, data) {
    // This would integrate with your socket.io or websocket system
    console.log(`[GameServer] Broadcasting ${eventType}:`, data);
  }

  /**
   * Get combat statistics
   * @returns {Object} Combat stats
   */
  getCombatStats() {
    return {
      totalMobs: this.mobs.size,
      aliveMobs: Array.from(this.mobs.values()).filter(m => !m.isDead).length,
      deadMobs: Array.from(this.mobs.values()).filter(m => m.isDead).length,
      totalPlayers: this.players.size,
      recentCombatEvents: this.combatLog.slice(-100)
    };
  }

  /**
   * Get all alive mobs
   * @returns {Array} Array of alive mobs
   */
  getAliveMobs() {
    return Array.from(this.mobs.values()).filter(m => !m.isDead);
  }

  /**
   * Register a player
   * @param {string} playerId - Player identifier
   * @param {Object} playerData - Player data
   */
  registerPlayer(playerId, playerData) {
    this.players.set(playerId, {
      id: playerId,
      health: 100,
      maxHealth: 100,
      armor: 0,
      armorPercent: 0,
      xp: 0,
      gold: 0,
      kills: 0,
      position: { x: 0, y: 0 },
      ...playerData
    });
    
    console.log(`[GameServer] Player ${playerId} registered`);
  }

  /**
   * Remove a player
   * @param {string} playerId - Player identifier
   */
  removePlayer(playerId) {
    this.players.delete(playerId);
    console.log(`[GameServer] Player ${playerId} removed`);
  }
}

// Singleton instance
let gameServerInstance = null;

/**
 * Get or create the game server instance
 * @returns {GameServer} The game server instance
 */
function getGameServer() {
  if (!gameServerInstance) {
    gameServerInstance = new GameServer();
    gameServerInstance.initialize();
  }
  return gameServerInstance;
}

module.exports = {
  GameServer,
  getGameServer
};
