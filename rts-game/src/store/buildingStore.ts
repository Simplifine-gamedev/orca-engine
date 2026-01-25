import { create } from 'zustand';
import { PlacedBuilding, BuildingType } from '../types/building';
import { buildingModels } from '../buildings/buildingModels';

interface BuildingState {
  buildings: PlacedBuilding[];
  selectedBuilding: PlacedBuilding | null;
  
  // Actions
  addBuilding: (type: BuildingType, position: { x: number; y: number }) => void;
  removeBuilding: (instanceId: string) => void;
  selectBuilding: (instanceId: string) => void;
  deselectBuilding: () => void;
  updateBuildingHealth: (instanceId: string, health: number) => void;
  updateConstructionProgress: (instanceId: string, progress: number) => void;
}

export const useBuildingStore = create<BuildingState>((set, get) => ({
  buildings: [],
  selectedBuilding: null,
  
  addBuilding: (type: BuildingType, position: { x: number; y: number }) => {
    const model = buildingModels[type];
    const newBuilding: PlacedBuilding = {
      ...model,
      instanceId: `${type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      position,
      health: model.hitPoints,
      productionQueue: [],
      constructionProgress: 0,
    };
    
    set((state) => ({
      buildings: [...state.buildings, newBuilding],
    }));
  },
  
  removeBuilding: (instanceId: string) => {
    set((state) => ({
      buildings: state.buildings.filter(b => b.instanceId !== instanceId),
      selectedBuilding: state.selectedBuilding?.instanceId === instanceId 
        ? null 
        : state.selectedBuilding,
    }));
  },
  
  selectBuilding: (instanceId: string) => {
    const building = get().buildings.find(b => b.instanceId === instanceId);
    if (building) {
      set({ selectedBuilding: building });
    }
  },
  
  deselectBuilding: () => {
    set({ selectedBuilding: null });
  },
  
  updateBuildingHealth: (instanceId: string, health: number) => {
    set((state) => ({
      buildings: state.buildings.map(b => 
        b.instanceId === instanceId ? { ...b, health } : b
      ),
    }));
  },
  
  updateConstructionProgress: (instanceId: string, progress: number) => {
    set((state) => ({
      buildings: state.buildings.map(b => 
        b.instanceId === instanceId ? { ...b, constructionProgress: progress } : b
      ),
    }));
  },
}));
