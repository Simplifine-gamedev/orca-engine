import { create } from 'zustand';
import { GameState, Resource } from '../types';

const initialResources: Resource[] = [
  {
    id: 'goldmine-1',
    type: 'goldmine',
    position: { x: 100, y: 100 },
    amountRemaining: 5000,
    maxAmount: 5000,
    workersAssigned: 0,
    gatherRate: 10
  },
  {
    id: 'goldmine-2',
    type: 'goldmine',
    position: { x: 300, y: 200 },
    amountRemaining: 3500,
    maxAmount: 5000,
    workersAssigned: 2,
    gatherRate: 10
  },
  {
    id: 'tree-1',
    type: 'tree',
    position: { x: 500, y: 150 },
    amountRemaining: 500,
    maxAmount: 500,
    workersAssigned: 1,
    gatherRate: 5
  }
];

export const useGameStore = create<GameState>((set) => ({
  resources: initialResources,
  selectedResourceId: null,
  
  selectResource: (id: string | null) => {
    set({ selectedResourceId: id });
  },
  
  addWorkerToResource: (id: string) => {
    set((state) => ({
      resources: state.resources.map((resource) =>
        resource.id === id
          ? { ...resource, workersAssigned: resource.workersAssigned + 1 }
          : resource
      )
    }));
  },
  
  removeWorkerFromResource: (id: string) => {
    set((state) => ({
      resources: state.resources.map((resource) =>
        resource.id === id && resource.workersAssigned > 0
          ? { ...resource, workersAssigned: resource.workersAssigned - 1 }
          : resource
      )
    }));
  }
}));
