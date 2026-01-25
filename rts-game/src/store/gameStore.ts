import { create } from 'zustand';
import { Unit, DamageEvent, DamageType, Position, GameSettings } from '../types';

interface GameState {
  units: Unit[];
  damageEvents: DamageEvent[];
  settings: GameSettings;
  
  // Actions
  addUnit: (unit: Omit<Unit, 'id' | 'lastAttackTime'>) => void;
  selectUnit: (id: string) => void;
  deselectAll: () => void;
  moveUnit: (id: string, position: Position) => void;
  setTarget: (attackerId: string, targetId: string) => void;
  dealDamage: (targetId: string, amount: number, type: DamageType, position: Position) => void;
  removeDamageEvent: (id: string) => void;
  updateUnits: (deltaTime: number) => void;
  toggleDamageNumbers: () => void;
  initializeGame: () => void;
}

const useGameStore = create<GameState>((set, get) => ({
  units: [],
  damageEvents: [],
  settings: {
    showDamageNumbers: true,
  },

  addUnit: (unit) => {
    const newUnit: Unit = {
      ...unit,
      id: `unit-${Date.now()}-${Math.random()}`,
      lastAttackTime: 0,
    };
    set((state) => ({ units: [...state.units, newUnit] }));
  },

  selectUnit: (id) => {
    set((state) => ({
      units: state.units.map((unit) => ({
        ...unit,
        isSelected: unit.id === id,
      })),
    }));
  },

  deselectAll: () => {
    set((state) => ({
      units: state.units.map((unit) => ({ ...unit, isSelected: false })),
    }));
  },

  moveUnit: (id, position) => {
    set((state) => ({
      units: state.units.map((unit) =>
        unit.id === id ? { ...unit, position, target: undefined } : unit
      ),
    }));
  },

  setTarget: (attackerId, targetId) => {
    set((state) => ({
      units: state.units.map((unit) =>
        unit.id === attackerId ? { ...unit, target: targetId } : unit
      ),
    }));
  },

  dealDamage: (targetId, amount, type, position) => {
    const damageEvent: DamageEvent = {
      id: `damage-${Date.now()}-${Math.random()}`,
      position,
      amount,
      type,
      timestamp: Date.now(),
    };

    set((state) => ({
      units: state.units.map((unit) => {
        if (unit.id === targetId) {
          const newHealth = Math.max(0, unit.health - amount);
          return { ...unit, health: newHealth };
        }
        return unit;
      }).filter(unit => unit.health > 0), // Remove dead units
      damageEvents: [...state.damageEvents, damageEvent],
    }));

    // Auto-remove damage event after animation
    setTimeout(() => {
      get().removeDamageEvent(damageEvent.id);
    }, 2000);
  },

  removeDamageEvent: (id) => {
    set((state) => ({
      damageEvents: state.damageEvents.filter((event) => event.id !== id),
    }));
  },

  updateUnits: (deltaTime) => {
    const state = get();
    const currentTime = Date.now();

    const updatedUnits = state.units.map((unit) => {
      if (!unit.target) return unit;

      const target = state.units.find((u) => u.id === unit.target);
      if (!target) {
        return { ...unit, target: undefined };
      }

      // Calculate distance to target
      const dx = target.position.x - unit.position.x;
      const dy = target.position.y - unit.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      // Attack range
      const attackRange = 50;

      if (distance <= attackRange) {
        // In range, try to attack
        const timeSinceLastAttack = currentTime - unit.lastAttackTime;
        const attackCooldown = 1000 / unit.attackSpeed; // Convert attacks per second to milliseconds

        if (timeSinceLastAttack >= attackCooldown) {
          // Perform attack
          const isCritical = Math.random() < 0.15; // 15% crit chance
          const damage = isCritical ? unit.attack * 2 : unit.attack;
          const damageType: DamageType = isCritical ? 'critical' : 'physical';

          // Deal damage on next tick to ensure state consistency
          setTimeout(() => {
            get().dealDamage(target.id, damage, damageType, target.position);
          }, 0);

          return { ...unit, lastAttackTime: currentTime };
        }
      } else {
        // Move towards target
        const moveDistance = unit.movementSpeed * (deltaTime / 1000);
        const moveX = (dx / distance) * moveDistance;
        const moveY = (dy / distance) * moveDistance;

        return {
          ...unit,
          position: {
            x: unit.position.x + moveX,
            y: unit.position.y + moveY,
          },
        };
      }

      return unit;
    });

    set({ units: updatedUnits });
  },

  toggleDamageNumbers: () => {
    set((state) => ({
      settings: {
        ...state.settings,
        showDamageNumbers: !state.settings.showDamageNumbers,
      },
    }));
  },

  initializeGame: () => {
    // Clear existing state
    set({ units: [], damageEvents: [] });

    // Create player units
    for (let i = 0; i < 3; i++) {
      get().addUnit({
        position: { x: 100 + i * 80, y: 300 },
        health: 100,
        maxHealth: 100,
        attack: 15,
        attackSpeed: 1.0,
        movementSpeed: 100,
        team: 'player',
        isSelected: false,
      });
    }

    // Create enemy units
    for (let i = 0; i < 3; i++) {
      get().addUnit({
        position: { x: 700 + i * 80, y: 300 },
        health: 100,
        maxHealth: 100,
        attack: 12,
        attackSpeed: 0.8,
        movementSpeed: 80,
        team: 'enemy',
        isSelected: false,
      });
    }
  },
}));

export default useGameStore;
