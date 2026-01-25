import { create } from 'zustand';
import { MobLairConfig } from '../objects/MobLair';

export interface Mob {
  id: string;
  type: string;
  lairId: string;
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  target?: { x: number; y: number };
  isAlive: boolean;
}

export interface MobStoreState {
  lairs: Map<string, MobLairConfig>;
  mobs: Map<string, Mob>;
  
  // Lair operations
  addLair: (lair: MobLairConfig) => void;
  removeLair: (lairId: string) => void;
  updateLair: (lairId: string, updates: Partial<MobLairConfig>) => void;
  destroyLair: (lairId: string) => void;
  getLair: (lairId: string) => MobLairConfig | undefined;
  getAllLairs: () => MobLairConfig[];
  
  // Mob operations
  spawnMob: (mob: Mob) => void;
  removeMob: (mobId: string) => void;
  updateMob: (mobId: string, updates: Partial<Mob>) => void;
  killMob: (mobId: string) => void;
  getMob: (mobId: string) => Mob | undefined;
  getAllMobs: () => Mob[];
  getMobsByLair: (lairId: string) => Mob[];
  getAliveMobs: () => Mob[];
  
  // Cleanup
  clearAll: () => void;
}

export const useMobStore = create<MobStoreState>((set, get) => ({
  lairs: new Map(),
  mobs: new Map(),
  
  // Lair operations
  addLair: (lair: MobLairConfig) => {
    set((state) => {
      const newLairs = new Map(state.lairs);
      newLairs.set(lair.id, lair);
      return { lairs: newLairs };
    });
  },
  
  removeLair: (lairId: string) => {
    set((state) => {
      const newLairs = new Map(state.lairs);
      newLairs.delete(lairId);
      
      // Also remove all mobs from this lair
      const newMobs = new Map(state.mobs);
      Array.from(newMobs.values())
        .filter(mob => mob.lairId === lairId)
        .forEach(mob => newMobs.delete(mob.id));
      
      return { lairs: newLairs, mobs: newMobs };
    });
  },
  
  updateLair: (lairId: string, updates: Partial<MobLairConfig>) => {
    set((state) => {
      const newLairs = new Map(state.lairs);
      const lair = newLairs.get(lairId);
      
      if (lair) {
        newLairs.set(lairId, { ...lair, ...updates });
      }
      
      return { lairs: newLairs };
    });
  },
  
  destroyLair: (lairId: string) => {
    get().updateLair(lairId, { 
      isDestroyed: true,
      health: 0,
    });
  },
  
  getLair: (lairId: string) => {
    return get().lairs.get(lairId);
  },
  
  getAllLairs: () => {
    return Array.from(get().lairs.values());
  },
  
  // Mob operations
  spawnMob: (mob: Mob) => {
    set((state) => {
      const newMobs = new Map(state.mobs);
      newMobs.set(mob.id, { ...mob, isAlive: true });
      return { mobs: newMobs };
    });
  },
  
  removeMob: (mobId: string) => {
    set((state) => {
      const newMobs = new Map(state.mobs);
      newMobs.delete(mobId);
      return { mobs: newMobs };
    });
  },
  
  updateMob: (mobId: string, updates: Partial<Mob>) => {
    set((state) => {
      const newMobs = new Map(state.mobs);
      const mob = newMobs.get(mobId);
      
      if (mob) {
        newMobs.set(mobId, { ...mob, ...updates });
      }
      
      return { mobs: newMobs };
    });
  },
  
  killMob: (mobId: string) => {
    get().updateMob(mobId, { isAlive: false, health: 0 });
    
    // Remove mob after delay for death animation
    setTimeout(() => {
      get().removeMob(mobId);
    }, 2000);
  },
  
  getMob: (mobId: string) => {
    return get().mobs.get(mobId);
  },
  
  getAllMobs: () => {
    return Array.from(get().mobs.values());
  },
  
  getMobsByLair: (lairId: string) => {
    return Array.from(get().mobs.values()).filter(
      mob => mob.lairId === lairId && mob.isAlive
    );
  },
  
  getAliveMobs: () => {
    return Array.from(get().mobs.values()).filter(mob => mob.isAlive);
  },
  
  clearAll: () => {
    set({ lairs: new Map(), mobs: new Map() });
  },
}));

// Selectors for performance
export const selectActiveLairs = (state: MobStoreState) =>
  Array.from(state.lairs.values()).filter(lair => !lair.isDestroyed);

export const selectMobCountByType = (state: MobStoreState) => {
  const counts: Record<string, number> = {};
  Array.from(state.mobs.values()).forEach(mob => {
    if (mob.isAlive) {
      counts[mob.type] = (counts[mob.type] || 0) + 1;
    }
  });
  return counts;
};

export const selectTotalAliveMobs = (state: MobStoreState) =>
  Array.from(state.mobs.values()).filter(mob => mob.isAlive).length;
