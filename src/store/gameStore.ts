import { create } from 'zustand';

export interface UnitQueueItem {
  id: string;
  unitType: string;
  timestamp: number;
  progress: number; // 0-100
}

export interface Building {
  id: string;
  type: string;
  position: { x: number; y: number };
  unitQueue: UnitQueueItem[];
}

interface GameState {
  buildings: Map<string, Building>;
  selectedBuildingId: string | null;
  
  // Actions
  selectBuilding: (buildingId: string | null) => void;
  trainUnit: (buildingId: string, unitType: string, count?: number) => void;
  cancelUnit: (buildingId: string, unitId: string) => void;
  updateUnitProgress: (buildingId: string, unitId: string, progress: number) => void;
  completeUnit: (buildingId: string, unitId: string) => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  buildings: new Map(),
  selectedBuildingId: null,

  selectBuilding: (buildingId) => {
    set({ selectedBuildingId: buildingId });
  },

  trainUnit: (buildingId, unitType, count = 1) => {
    const building = get().buildings.get(buildingId);
    if (!building) return;

    const newUnits: UnitQueueItem[] = [];
    for (let i = 0; i < count; i++) {
      newUnits.push({
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        unitType,
        timestamp: Date.now() + i,
        progress: 0,
      });
    }

    const updatedBuilding: Building = {
      ...building,
      unitQueue: [...building.unitQueue, ...newUnits],
    };

    const updatedBuildings = new Map(get().buildings);
    updatedBuildings.set(buildingId, updatedBuilding);

    set({ buildings: updatedBuildings });
  },

  cancelUnit: (buildingId, unitId) => {
    const building = get().buildings.get(buildingId);
    if (!building) return;

    const updatedBuilding: Building = {
      ...building,
      unitQueue: building.unitQueue.filter((unit) => unit.id !== unitId),
    };

    const updatedBuildings = new Map(get().buildings);
    updatedBuildings.set(buildingId, updatedBuilding);

    set({ buildings: updatedBuildings });
  },

  updateUnitProgress: (buildingId, unitId, progress) => {
    const building = get().buildings.get(buildingId);
    if (!building) return;

    const updatedBuilding: Building = {
      ...building,
      unitQueue: building.unitQueue.map((unit) =>
        unit.id === unitId ? { ...unit, progress } : unit
      ),
    };

    const updatedBuildings = new Map(get().buildings);
    updatedBuildings.set(buildingId, updatedBuilding);

    set({ buildings: updatedBuildings });
  },

  completeUnit: (buildingId, unitId) => {
    const building = get().buildings.get(buildingId);
    if (!building) return;

    const updatedBuilding: Building = {
      ...building,
      unitQueue: building.unitQueue.filter((unit) => unit.id !== unitId),
    };

    const updatedBuildings = new Map(get().buildings);
    updatedBuildings.set(buildingId, updatedBuilding);

    set({ buildings: updatedBuildings });
    
    // Here you would spawn the actual unit in the game
    console.log(`Unit ${unitId} of type ${building.unitQueue.find(u => u.id === unitId)?.unitType} completed!`);
  },
}));
