import { create } from 'zustand';

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Worker {
  id: string;
  position: Vector3;
  state: 'idle' | 'mining' | 'building' | 'moving';
  targetResource?: string;
  carrying: number;
}

export interface Resource {
  id: string;
  type: 'gold' | 'wood';
  position: Vector3;
  amount: number;
}

export interface Building {
  id: string;
  type: 'townhall' | 'barracks' | 'farm';
  position: Vector3;
  isConstructed: boolean;
}

export interface BuildingPlacement {
  type: 'townhall' | 'barracks' | 'farm' | null;
  isActive: boolean;
  ghostPosition: Vector3 | null;
}

interface GameState {
  workers: Worker[];
  resources: Resource[];
  buildings: Building[];
  selectedWorker: string | null;
  buildingPlacement: BuildingPlacement;
  
  // Worker actions
  addWorker: (worker: Worker) => void;
  selectWorker: (id: string | null) => void;
  setWorkerState: (id: string, state: Worker['state']) => void;
  updateWorkerPosition: (id: string, position: Vector3) => void;
  
  // Building placement actions
  startBuildingPlacement: (type: 'townhall' | 'barracks' | 'farm') => void;
  updateBuildingGhostPosition: (position: Vector3 | null) => void;
  cancelBuildingPlacement: () => void;
  confirmBuildingPlacement: () => void;
  
  // Resource actions
  addResource: (resource: Resource) => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  workers: [],
  resources: [],
  buildings: [],
  selectedWorker: null,
  buildingPlacement: {
    type: null,
    isActive: false,
    ghostPosition: null,
  },
  
  addWorker: (worker) => set((state) => ({
    workers: [...state.workers, worker],
  })),
  
  selectWorker: (id) => set({ selectedWorker: id }),
  
  setWorkerState: (id, workerState) => set((state) => ({
    workers: state.workers.map((w) =>
      w.id === id ? { ...w, state: workerState } : w
    ),
  })),
  
  updateWorkerPosition: (id, position) => set((state) => ({
    workers: state.workers.map((w) =>
      w.id === id ? { ...w, position } : w
    ),
  })),
  
  startBuildingPlacement: (type) => set({
    buildingPlacement: {
      type,
      isActive: true,
      ghostPosition: null,
    },
  }),
  
  updateBuildingGhostPosition: (position) => set((state) => ({
    buildingPlacement: {
      ...state.buildingPlacement,
      ghostPosition: position,
    },
  })),
  
  cancelBuildingPlacement: () => set({
    buildingPlacement: {
      type: null,
      isActive: false,
      ghostPosition: null,
    },
  }),
  
  confirmBuildingPlacement: () => {
    const state = get();
    const { buildingPlacement } = state;
    
    if (buildingPlacement.isActive && buildingPlacement.ghostPosition && buildingPlacement.type) {
      const newBuilding: Building = {
        id: `building-${Date.now()}`,
        type: buildingPlacement.type,
        position: buildingPlacement.ghostPosition,
        isConstructed: false,
      };
      
      set((state) => ({
        buildings: [...state.buildings, newBuilding],
        buildingPlacement: {
          type: null,
          isActive: false,
          ghostPosition: null,
        },
      }));
    }
  },
  
  addResource: (resource) => set((state) => ({
    resources: [...state.resources, resource],
  })),
}));
