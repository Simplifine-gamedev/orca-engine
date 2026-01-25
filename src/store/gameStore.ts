import { create } from 'zustand';
import { ResourceData, SelectableEntity } from '../types/resource';

interface GameState {
  // Selection state
  selectedEntity: SelectableEntity | null;
  
  // Resources in the game
  resources: ResourceData[];
  
  // Actions
  selectEntity: (entity: SelectableEntity | null) => void;
  deselectEntity: () => void;
  addResource: (resource: ResourceData) => void;
  updateResource: (id: string, updates: Partial<ResourceData>) => void;
  assignWorker: (resourceId: string) => void;
  unassignWorker: (resourceId: string) => void;
}

export const useGameStore = create<GameState>((set) => ({
  // Initial state
  selectedEntity: null,
  resources: [],
  
  // Actions
  selectEntity: (entity) => set({ selectedEntity: entity }),
  
  deselectEntity: () => set({ selectedEntity: null }),
  
  addResource: (resource) => set((state) => ({
    resources: [...state.resources, resource]
  })),
  
  updateResource: (id, updates) => set((state) => ({
    resources: state.resources.map((res) =>
      res.id === id ? { ...res, ...updates } : res
    )
  })),
  
  assignWorker: (resourceId) => set((state) => ({
    resources: state.resources.map((res) =>
      res.id === resourceId && res.workersAssigned < res.maxWorkers
        ? { ...res, workersAssigned: res.workersAssigned + 1 }
        : res
    )
  })),
  
  unassignWorker: (resourceId) => set((state) => ({
    resources: state.resources.map((res) =>
      res.id === resourceId && res.workersAssigned > 0
        ? { ...res, workersAssigned: res.workersAssigned - 1 }
        : res
    )
  })),
}));
