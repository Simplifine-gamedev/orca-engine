import { create } from 'zustand';
import { GameState, Resources, Building, Worker } from '../types';

interface GameStore extends GameState {
  // Resource actions
  addWood: (amount: number) => void;
  addGold: (amount: number) => void;
  addStone: (amount: number) => void;
  addFood: (amount: number) => void;
  spendResources: (cost: Partial<Resources>) => boolean;
  
  // Worker actions
  addWorker: (worker: Worker) => void;
  removeWorker: (workerId: string) => void;
  updateWorker: (workerId: string, updates: Partial<Worker>) => void;
  
  // Building actions
  addBuilding: (building: Building) => void;
  removeBuilding: (buildingId: string) => void;
  
  // Game control
  togglePause: () => void;
  setGameSpeed: (speed: number) => void;
  resetGame: () => void;
}

const initialResources: Resources = {
  wood: 100,
  gold: 100,
  stone: 50,
  food: 200,
};

export const useGameStore = create<GameStore>((set, get) => ({
  // Initial state
  resources: initialResources,
  trees: [],
  workers: [],
  buildings: [],
  gameSpeed: 1,
  isPaused: false,

  // Resource actions
  addWood: (amount: number) =>
    set((state) => ({
      resources: {
        ...state.resources,
        wood: state.resources.wood + amount,
      },
    })),

  addGold: (amount: number) =>
    set((state) => ({
      resources: {
        ...state.resources,
        gold: state.resources.gold + amount,
      },
    })),

  addStone: (amount: number) =>
    set((state) => ({
      resources: {
        ...state.resources,
        stone: state.resources.stone + amount,
      },
    })),

  addFood: (amount: number) =>
    set((state) => ({
      resources: {
        ...state.resources,
        food: state.resources.food + amount,
      },
    })),

  spendResources: (cost: Partial<Resources>): boolean => {
    const current = get().resources;
    
    // Check if we have enough resources
    const canAfford = Object.entries(cost).every(
      ([resource, amount]) => current[resource as keyof Resources] >= (amount || 0)
    );

    if (canAfford) {
      set((state) => ({
        resources: {
          wood: state.resources.wood - (cost.wood || 0),
          gold: state.resources.gold - (cost.gold || 0),
          stone: state.resources.stone - (cost.stone || 0),
          food: state.resources.food - (cost.food || 0),
        },
      }));
      return true;
    }
    
    return false;
  },

  // Worker actions
  addWorker: (worker: Worker) =>
    set((state) => ({
      workers: [...state.workers, worker],
    })),

  removeWorker: (workerId: string) =>
    set((state) => ({
      workers: state.workers.filter((w) => w.id !== workerId),
    })),

  updateWorker: (workerId: string, updates: Partial<Worker>) =>
    set((state) => ({
      workers: state.workers.map((w) =>
        w.id === workerId ? { ...w, ...updates } : w
      ),
    })),

  // Building actions
  addBuilding: (building: Building) =>
    set((state) => ({
      buildings: [...state.buildings, building],
    })),

  removeBuilding: (buildingId: string) =>
    set((state) => ({
      buildings: state.buildings.filter((b) => b.id !== buildingId),
    })),

  // Game control
  togglePause: () =>
    set((state) => ({
      isPaused: !state.isPaused,
    })),

  setGameSpeed: (speed: number) =>
    set({ gameSpeed: speed }),

  resetGame: () =>
    set({
      resources: initialResources,
      trees: [],
      workers: [],
      buildings: [],
      gameSpeed: 1,
      isPaused: false,
    }),
}));
