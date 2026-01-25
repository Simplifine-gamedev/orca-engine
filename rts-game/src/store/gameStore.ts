import { create } from 'zustand';
import { DamageEvent, DamageType, Unit, GameSettings } from '../types';

interface GameState {
  // Units
  units: Unit[];
  selectedUnit: Unit | null;
  
  // Damage events
  damageEvents: DamageEvent[];
  
  // Settings
  settings: GameSettings;
  
  // Actions
  addUnit: (unit: Unit) => void;
  removeUnit: (id: string) => void;
  selectUnit: (unit: Unit | null) => void;
  updateUnitHealth: (id: string, health: number) => void;
  
  // Damage event actions
  emitDamage: (amount: number, type: DamageType, x: number, y: number) => void;
  removeDamageEvent: (id: string) => void;
  
  // Settings actions
  toggleDamageNumbers: () => void;
  updateSettings: (settings: Partial<GameSettings>) => void;
  
  // Game actions
  attackUnit: (attackerId: string, targetId: string) => void;
}

const createDamageEvent = (
  amount: number,
  type: DamageType,
  x: number,
  y: number
): DamageEvent => ({
  id: `${Date.now()}-${Math.random()}`,
  amount,
  type,
  x,
  y,
  timestamp: Date.now(),
});

export const useGameStore = create<GameState>((set, get) => ({
  // Initial state
  units: [],
  selectedUnit: null,
  damageEvents: [],
  settings: {
    showDamageNumbers: true,
    soundEnabled: true,
    musicVolume: 0.5,
  },

  // Unit actions
  addUnit: (unit) => set((state) => ({
    units: [...state.units, unit],
  })),

  removeUnit: (id) => set((state) => ({
    units: state.units.filter((u) => u.id !== id),
  })),

  selectUnit: (unit) => set({ selectedUnit: unit }),

  updateUnitHealth: (id, health) => set((state) => ({
    units: state.units.map((u) =>
      u.id === id ? { ...u, health: Math.max(0, Math.min(u.maxHealth, health)) } : u
    ),
  })),

  // Damage event actions
  emitDamage: (amount, type, x, y) => {
    const { settings } = get();
    if (!settings.showDamageNumbers) return;

    const event = createDamageEvent(amount, type, x, y);
    set((state) => ({
      damageEvents: [...state.damageEvents, event],
    }));
  },

  removeDamageEvent: (id) => set((state) => ({
    damageEvents: state.damageEvents.filter((e) => e.id !== id),
  })),

  // Settings actions
  toggleDamageNumbers: () => set((state) => ({
    settings: {
      ...state.settings,
      showDamageNumbers: !state.settings.showDamageNumbers,
    },
  })),

  updateSettings: (newSettings) => set((state) => ({
    settings: {
      ...state.settings,
      ...newSettings,
    },
  })),

  // Game actions
  attackUnit: (attackerId, targetId) => {
    const { units, emitDamage, updateUnitHealth } = get();
    const attacker = units.find((u) => u.id === attackerId);
    const target = units.find((u) => u.id === targetId);

    if (!attacker || !target) return;

    // Calculate damage (could be more complex with armor, resistances, etc.)
    const damage = attacker.attack;
    const newHealth = target.health - damage;

    // Emit damage event at target's position
    emitDamage(damage, 'physical', target.x, target.y);

    // Update target's health
    updateUnitHealth(targetId, newHealth);

    // Remove unit if dead
    if (newHealth <= 0) {
      setTimeout(() => {
        get().removeUnit(targetId);
      }, 500);
    }
  },
}));
