import { create } from 'zustand';
import { MobLairConfig, LairType, LAIR_CONFIGS, LootItem } from '../objects/MobLair';

export interface Mob {
  id: string;
  type: string;
  level: number;
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  lairId: string | null;
  spawnTime: number;
}

export interface LairSpawnTimer {
  lairId: string;
  nextSpawnTime: number;
  currentMobCount: number;
}

interface MobStoreState {
  mobs: Record<string, Mob>;
  lairs: Record<string, MobLairConfig>;
  lairSpawnTimers: Record<string, LairSpawnTimer>;
  isRunning: boolean;

  // Lair actions
  createLair: (type: LairType, position: { x: number; y: number }) => string;
  destroyLair: (lairId: string) => LootItem[];
  damageLair: (lairId: string, damage: number) => boolean;
  getLair: (lairId: string) => MobLairConfig | undefined;
  getAllLairs: () => MobLairConfig[];

  // Mob actions
  spawnMob: (lairId: string) => string | null;
  killMob: (mobId: string) => void;
  getMob: (mobId: string) => Mob | undefined;
  getMobsForLair: (lairId: string) => Mob[];
  getAllMobs: () => Mob[];

  // Spawning system
  startSpawning: () => void;
  stopSpawning: () => void;
  updateSpawning: (deltaTime: number) => void;

  // Utility
  reset: () => void;
}

let nextLairId = 1;
let nextMobId = 1;
let spawnInterval: NodeJS.Timeout | null = null;

const generateLairId = (): string => `lair_${nextLairId++}`;
const generateMobId = (): string => `mob_${nextMobId++}`;

const randomInRange = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

const getRandomPositionInRadius = (
  center: { x: number; y: number },
  radius: number
): { x: number; y: number } => {
  const angle = Math.random() * Math.PI * 2;
  const distance = Math.random() * radius;
  return {
    x: center.x + Math.cos(angle) * distance,
    y: center.y + Math.sin(angle) * distance,
  };
};

const calculateMobHealth = (type: string, level: number): number => {
  // Base health by mob type
  const baseHealth: Record<string, number> = {
    goblin_warrior: 50,
    goblin_archer: 40,
    ogre: 200,
    cave_troll: 150,
    wolf: 60,
    dire_wolf: 100,
    bandit: 70,
    bandit_rogue: 80,
    bandit_chief: 120,
    skeleton: 50,
    zombie: 80,
    ghoul: 110,
  };

  return (baseHealth[type] || 50) * level;
};

const rollLoot = (lootTable: LootItem[]): LootItem[] => {
  const droppedLoot: LootItem[] = [];

  for (const lootEntry of lootTable) {
    if (Math.random() < lootEntry.chance) {
      const quantity = randomInRange(
        lootEntry.quantity.min,
        lootEntry.quantity.max
      );
      droppedLoot.push({
        ...lootEntry,
        quantity: { min: quantity, max: quantity },
      });
    }
  }

  return droppedLoot;
};

export const useMobStore = create<MobStoreState>((set, get) => ({
  mobs: {},
  lairs: {},
  lairSpawnTimers: {},
  isRunning: false,

  createLair: (type: LairType, position: { x: number; y: number }) => {
    const lairId = generateLairId();
    const config = LAIR_CONFIGS[type];

    const newLair: MobLairConfig = {
      id: lairId,
      type,
      position,
      health: config.maxHealth,
      maxHealth: config.maxHealth,
      spawnInterval: config.spawnInterval,
      maxMobs: config.maxMobs,
      spawnRadius: config.spawnRadius,
      lootTable: config.lootTable,
      destroyed: false,
    };

    set((state) => ({
      lairs: { ...state.lairs, [lairId]: newLair },
      lairSpawnTimers: {
        ...state.lairSpawnTimers,
        [lairId]: {
          lairId,
          nextSpawnTime: Date.now() + config.spawnInterval,
          currentMobCount: 0,
        },
      },
    }));

    return lairId;
  },

  destroyLair: (lairId: string) => {
    const state = get();
    const lair = state.lairs[lairId];

    if (!lair || lair.destroyed) {
      return [];
    }

    // Roll for loot
    const droppedLoot = lair.lootTable ? rollLoot(lair.lootTable) : [];

    // Mark lair as destroyed
    set((state) => ({
      lairs: {
        ...state.lairs,
        [lairId]: { ...lair, destroyed: true, health: 0 },
      },
    }));

    // Kill all mobs from this lair
    const mobsToKill = Object.values(state.mobs)
      .filter((mob) => mob.lairId === lairId)
      .map((mob) => mob.id);

    mobsToKill.forEach((mobId) => get().killMob(mobId));

    // Remove spawn timer
    set((state) => {
      const newTimers = { ...state.lairSpawnTimers };
      delete newTimers[lairId];
      return { lairSpawnTimers: newTimers };
    });

    return droppedLoot;
  },

  damageLair: (lairId: string, damage: number) => {
    const state = get();
    const lair = state.lairs[lairId];

    if (!lair || lair.destroyed) {
      return false;
    }

    const newHealth = Math.max(0, lair.health - damage);
    const destroyed = newHealth <= 0;

    set((state) => ({
      lairs: {
        ...state.lairs,
        [lairId]: { ...lair, health: newHealth, destroyed },
      },
    }));

    if (destroyed) {
      get().destroyLair(lairId);
    }

    return destroyed;
  },

  getLair: (lairId: string) => {
    return get().lairs[lairId];
  },

  getAllLairs: () => {
    return Object.values(get().lairs).filter((lair) => !lair.destroyed);
  },

  spawnMob: (lairId: string) => {
    const state = get();
    const lair = state.lairs[lairId];
    const timer = state.lairSpawnTimers[lairId];

    if (!lair || lair.destroyed || !timer) {
      return null;
    }

    // Check if we've reached max mob count for this lair
    const currentMobs = get().getMobsForLair(lairId);
    if (currentMobs.length >= lair.maxMobs) {
      return null;
    }

    // Get lair configuration
    const config = LAIR_CONFIGS[lair.type];
    const mobSpawns = config.mobSpawns;

    // Select a random mob type to spawn from the lair's spawn list
    const spawnConfig = mobSpawns[Math.floor(Math.random() * mobSpawns.length)];

    // Generate mob
    const mobId = generateMobId();
    const level = randomInRange(spawnConfig.level.min, spawnConfig.level.max);
    const position = getRandomPositionInRadius(lair.position, lair.spawnRadius);
    const maxHealth = calculateMobHealth(spawnConfig.mobType, level);

    const newMob: Mob = {
      id: mobId,
      type: spawnConfig.mobType,
      level,
      position,
      health: maxHealth,
      maxHealth,
      lairId,
      spawnTime: Date.now(),
    };

    set((state) => ({
      mobs: { ...state.mobs, [mobId]: newMob },
      lairSpawnTimers: {
        ...state.lairSpawnTimers,
        [lairId]: {
          ...timer,
          currentMobCount: timer.currentMobCount + 1,
          nextSpawnTime: Date.now() + lair.spawnInterval,
        },
      },
    }));

    return mobId;
  },

  killMob: (mobId: string) => {
    set((state) => {
      const mob = state.mobs[mobId];
      if (!mob) return state;

      const newMobs = { ...state.mobs };
      delete newMobs[mobId];

      // Update lair spawn timer count
      const newTimers = { ...state.lairSpawnTimers };
      if (mob.lairId && newTimers[mob.lairId]) {
        newTimers[mob.lairId] = {
          ...newTimers[mob.lairId],
          currentMobCount: Math.max(0, newTimers[mob.lairId].currentMobCount - 1),
        };
      }

      return {
        mobs: newMobs,
        lairSpawnTimers: newTimers,
      };
    });
  },

  getMob: (mobId: string) => {
    return get().mobs[mobId];
  },

  getMobsForLair: (lairId: string) => {
    return Object.values(get().mobs).filter((mob) => mob.lairId === lairId);
  },

  getAllMobs: () => {
    return Object.values(get().mobs);
  },

  startSpawning: () => {
    if (get().isRunning) {
      return;
    }

    set({ isRunning: true });

    // Start the spawning interval
    spawnInterval = setInterval(() => {
      const state = get();
      const now = Date.now();

      // Check each lair's spawn timer
      Object.keys(state.lairs).forEach((lairId) => {
        const lair = state.lairs[lairId];
        const timer = state.lairSpawnTimers[lairId];

        if (lair && !lair.destroyed && timer && now >= timer.nextSpawnTime) {
          // Attempt to spawn mob
          get().spawnMob(lairId);
        }
      });
    }, 1000); // Check every second
  },

  stopSpawning: () => {
    if (spawnInterval) {
      clearInterval(spawnInterval);
      spawnInterval = null;
    }
    set({ isRunning: false });
  },

  updateSpawning: (deltaTime: number) => {
    const state = get();
    const now = Date.now();

    // Manual update for each lair
    Object.keys(state.lairs).forEach((lairId) => {
      const lair = state.lairs[lairId];
      const timer = state.lairSpawnTimers[lairId];

      if (lair && !lair.destroyed && timer && now >= timer.nextSpawnTime) {
        get().spawnMob(lairId);
      }
    });
  },

  reset: () => {
    get().stopSpawning();
    set({
      mobs: {},
      lairs: {},
      lairSpawnTimers: {},
      isRunning: false,
    });
    nextLairId = 1;
    nextMobId = 1;
  },
}));

export default useMobStore;
