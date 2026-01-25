/**
 * Mob Store - Configuration for all mob types in Orca RTS
 * Manages mob health, armor, and combat stats
 */

export interface MobStats {
  health: number;
  maxHealth: number;
  armor: number; // Flat damage reduction
  armorPercent: number; // Percentage damage reduction (0-1)
  damage: number;
  attackSpeed: number; // Attacks per second
  moveSpeed: number;
  attackRange: number;
  xpReward: number;
  goldReward: number;
}

export interface MobConfig {
  id: string;
  name: string;
  type: 'melee' | 'ranged' | 'elite' | 'boss';
  stats: MobStats;
  description: string;
}

/**
 * Mob Configurations
 * 
 * Balance Notes:
 * - Increased health values to prevent one-shotting
 * - Added armor system for damage mitigation
 * - Heavy soldiers deal ~40-60 damage per hit
 * - Mobs should survive at least 3-5 hits from heavy soldiers
 */
export const MOB_CONFIGS: Record<string, MobConfig> = {
  goblin: {
    id: 'goblin',
    name: 'Goblin',
    type: 'melee',
    stats: {
      health: 150, // Increased from ~30-50 to survive multiple hits
      maxHealth: 150,
      armor: 5, // Reduces damage by 5 flat
      armorPercent: 0.1, // Reduces damage by 10%
      damage: 15,
      attackSpeed: 1.2,
      moveSpeed: 3.5,
      attackRange: 1.5,
      xpReward: 25,
      goldReward: 10
    },
    description: 'Fast but lightly armored melee fighter'
  },

  orc: {
    id: 'orc',
    name: 'Orc Warrior',
    type: 'melee',
    stats: {
      health: 250, // Increased from ~60-80 to provide challenge
      maxHealth: 250,
      armor: 10, // Moderate armor
      armorPercent: 0.15, // 15% damage reduction
      damage: 25,
      attackSpeed: 0.9,
      moveSpeed: 2.8,
      attackRange: 2.0,
      xpReward: 40,
      goldReward: 20
    },
    description: 'Tough melee fighter with moderate armor'
  },

  orc_archer: {
    id: 'orc_archer',
    name: 'Orc Archer',
    type: 'ranged',
    stats: {
      health: 120, // Lower health but ranged advantage
      maxHealth: 120,
      armor: 3,
      armorPercent: 0.05, // Light armor
      damage: 20,
      attackSpeed: 1.5,
      moveSpeed: 3.0,
      attackRange: 8.0,
      xpReward: 35,
      goldReward: 15
    },
    description: 'Ranged attacker with lower health'
  },

  orc_berserker: {
    id: 'orc_berserker',
    name: 'Orc Berserker',
    type: 'elite',
    stats: {
      health: 400, // Elite unit - much tankier
      maxHealth: 400,
      armor: 15,
      armorPercent: 0.20, // 20% damage reduction
      damage: 45,
      attackSpeed: 1.1,
      moveSpeed: 3.2,
      attackRange: 2.0,
      xpReward: 75,
      goldReward: 40
    },
    description: 'Elite warrior with high health and damage'
  },

  troll: {
    id: 'troll',
    name: 'Troll',
    type: 'elite',
    stats: {
      health: 500, // Very tanky
      maxHealth: 500,
      armor: 20,
      armorPercent: 0.25, // 25% damage reduction
      damage: 50,
      attackSpeed: 0.7,
      moveSpeed: 2.2,
      attackRange: 2.5,
      xpReward: 100,
      goldReward: 50
    },
    description: 'Massive creature with regeneration and high armor'
  },

  goblin_chief: {
    id: 'goblin_chief',
    name: 'Goblin Chief',
    type: 'boss',
    stats: {
      health: 800, // Boss-level health
      maxHealth: 800,
      armor: 25,
      armorPercent: 0.30, // 30% damage reduction
      damage: 60,
      attackSpeed: 1.0,
      moveSpeed: 3.0,
      attackRange: 3.0,
      xpReward: 200,
      goldReward: 100
    },
    description: 'Boss-level goblin leader'
  },

  orc_warlord: {
    id: 'orc_warlord',
    name: 'Orc Warlord',
    type: 'boss',
    stats: {
      health: 1200, // Major boss
      maxHealth: 1200,
      armor: 35,
      armorPercent: 0.35, // 35% damage reduction
      damage: 80,
      attackSpeed: 0.8,
      moveSpeed: 2.5,
      attackRange: 3.0,
      xpReward: 300,
      goldReward: 150
    },
    description: 'Powerful orc commander with heavy armor'
  }
};

/**
 * Calculate actual damage after armor mitigation
 * @param baseDamage - Raw damage before armor
 * @param armor - Flat armor value
 * @param armorPercent - Percentage armor (0-1)
 * @returns Actual damage dealt
 */
export function calculateDamage(baseDamage: number, armor: number, armorPercent: number): number {
  // Apply flat armor reduction first
  let damage = Math.max(0, baseDamage - armor);
  
  // Then apply percentage reduction
  damage = damage * (1 - armorPercent);
  
  // Ensure minimum damage of 1 if base damage > 0
  return baseDamage > 0 ? Math.max(1, Math.floor(damage)) : 0;
}

/**
 * Get mob configuration by ID
 * @param mobId - The mob identifier
 * @returns Mob configuration or undefined
 */
export function getMobConfig(mobId: string): MobConfig | undefined {
  return MOB_CONFIGS[mobId];
}

/**
 * Create a new mob instance with full health
 * @param mobId - The mob identifier
 * @returns Fresh mob stats or undefined
 */
export function createMobInstance(mobId: string): MobStats | undefined {
  const config = getMobConfig(mobId);
  if (!config) return undefined;
  
  return {
    ...config.stats,
    health: config.stats.maxHealth // Start at full health
  };
}

/**
 * Get all mob types
 * @returns Array of all mob configurations
 */
export function getAllMobs(): MobConfig[] {
  return Object.values(MOB_CONFIGS);
}

/**
 * Get mobs by type
 * @param type - Mob type to filter by
 * @returns Array of matching mob configurations
 */
export function getMobsByType(type: MobConfig['type']): MobConfig[] {
  return Object.values(MOB_CONFIGS).filter(mob => mob.type === type);
}

export default {
  MOB_CONFIGS,
  getMobConfig,
  createMobInstance,
  calculateDamage,
  getAllMobs,
  getMobsByType
};
