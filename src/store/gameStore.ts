import { create } from 'zustand';
import { Unit, Building, AnimationState } from '../types/unit';

interface GameState {
  units: Map<string, Unit>;
  buildings: Map<string, Building>;
  spawnUnit: (buildingId: string, unitType: string) => void;
  updateUnitAnimation: (unitId: string, state: AnimationState) => void;
  removeUnit: (unitId: string) => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  units: new Map(),
  buildings: new Map(),

  spawnUnit: (buildingId: string, unitType: string) => {
    const building = get().buildings.get(buildingId);
    if (!building) return;

    const unitId = `unit_${Date.now()}_${Math.random()}`;
    const newUnit: Unit = {
      id: unitId,
      position: { ...building.spawnPoint },
      type: unitType,
      health: 100,
      maxHealth: 100,
      // FIX: Initialize with 'spawning' animation state instead of undefined
      // This prevents T-pose by ensuring animation state is set from the start
      animationState: 'spawning',
      isSpawning: true,
    };

    set((state) => {
      const newUnits = new Map(state.units);
      newUnits.set(unitId, newUnit);
      return { units: newUnits };
    });

    // Transition from spawning to idle after spawn animation completes
    // This gives time for the spawn animation to play before going idle
    setTimeout(() => {
      get().updateUnitAnimation(unitId, 'idle');
      set((state) => {
        const newUnits = new Map(state.units);
        const unit = newUnits.get(unitId);
        if (unit) {
          newUnits.set(unitId, { ...unit, isSpawning: false });
        }
        return { units: newUnits };
      });
    }, 500); // 500ms spawn animation duration
  },

  updateUnitAnimation: (unitId: string, animationState: AnimationState) => {
    set((state) => {
      const newUnits = new Map(state.units);
      const unit = newUnits.get(unitId);
      if (unit) {
        newUnits.set(unitId, { ...unit, animationState });
      }
      return { units: newUnits };
    });
  },

  removeUnit: (unitId: string) => {
    set((state) => {
      const newUnits = new Map(state.units);
      newUnits.delete(unitId);
      return { units: newUnits };
    });
  },
}));
