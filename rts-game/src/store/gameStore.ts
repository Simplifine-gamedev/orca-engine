import { create } from 'zustand';
import { GameState, Unit, Position } from '../types';

interface GameStore extends GameState {
  // Actions
  selectUnit: (unitId: string) => void;
  selectMultipleUnits: (unitIds: string[]) => void;
  deselectAllUnits: () => void;
  selectAllIdleWorkers: () => void;
  getIdleWorkers: () => Unit[];
  getIdleWorkerCount: () => number;
  moveUnit: (unitId: string, position: Position) => void;
  setUnitTask: (unitId: string, task: string) => void;
  updateUnit: (unitId: string, updates: Partial<Unit>) => void;
}

export const useGameStore = create<GameStore>((set, get) => ({
  // Initial state
  units: [
    {
      id: '1',
      type: 'worker',
      position: { x: 100, y: 100 },
      health: 100,
      maxHealth: 100,
      isSelected: false,
      isIdle: true,
    },
    {
      id: '2',
      type: 'worker',
      position: { x: 150, y: 120 },
      health: 100,
      maxHealth: 100,
      isSelected: false,
      isIdle: true,
    },
    {
      id: '3',
      type: 'worker',
      position: { x: 200, y: 150 },
      health: 100,
      maxHealth: 100,
      isSelected: false,
      isIdle: false,
      currentTask: 'gathering',
    },
    {
      id: '4',
      type: 'soldier',
      position: { x: 300, y: 200 },
      health: 150,
      maxHealth: 150,
      isSelected: false,
      isIdle: true,
    },
    {
      id: '5',
      type: 'worker',
      position: { x: 180, y: 180 },
      health: 100,
      maxHealth: 100,
      isSelected: false,
      isIdle: true,
    },
  ],
  buildings: [
    {
      id: 'b1',
      type: 'base',
      position: { x: 400, y: 400 },
      health: 500,
      maxHealth: 500,
    },
  ],
  resources: {
    wood: 200,
    gold: 150,
    stone: 100,
  },
  selectedUnits: [],

  // Actions
  selectUnit: (unitId) =>
    set((state) => ({
      units: state.units.map((unit) =>
        unit.id === unitId ? { ...unit, isSelected: true } : unit
      ),
      selectedUnits: [unitId],
    })),

  selectMultipleUnits: (unitIds) =>
    set((state) => ({
      units: state.units.map((unit) => ({
        ...unit,
        isSelected: unitIds.includes(unit.id),
      })),
      selectedUnits: unitIds,
    })),

  deselectAllUnits: () =>
    set((state) => ({
      units: state.units.map((unit) => ({ ...unit, isSelected: false })),
      selectedUnits: [],
    })),

  getIdleWorkers: () => {
    const state = get();
    return state.units.filter((unit) => unit.type === 'worker' && unit.isIdle);
  },

  getIdleWorkerCount: () => {
    const idleWorkers = get().getIdleWorkers();
    return idleWorkers.length;
  },

  selectAllIdleWorkers: () => {
    const idleWorkers = get().getIdleWorkers();
    const idleWorkerIds = idleWorkers.map((worker) => worker.id);
    
    set((state) => ({
      units: state.units.map((unit) => ({
        ...unit,
        isSelected: idleWorkerIds.includes(unit.id),
      })),
      selectedUnits: idleWorkerIds,
    }));
  },

  moveUnit: (unitId, position) =>
    set((state) => ({
      units: state.units.map((unit) =>
        unit.id === unitId ? { ...unit, position } : unit
      ),
    })),

  setUnitTask: (unitId, task) =>
    set((state) => ({
      units: state.units.map((unit) =>
        unit.id === unitId
          ? { ...unit, currentTask: task, isIdle: false }
          : unit
      ),
    })),

  updateUnit: (unitId, updates) =>
    set((state) => ({
      units: state.units.map((unit) =>
        unit.id === unitId ? { ...unit, ...updates } : unit
      ),
    })),
}));
