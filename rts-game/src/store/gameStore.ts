import { create } from 'zustand';
import { GameState, Unit, Building } from '../types';

interface GameStore extends GameState {
  selectBuilding: (id: string | null) => void;
  garrisonUnit: (unitId: string, buildingId: string) => void;
  releaseUnit: (unitId: string, buildingId: string) => void;
  releaseAllUnits: (buildingId: string) => void;
}

export const useGameStore = create<GameStore>((set) => ({
  units: {
    'unit-1': {
      id: 'unit-1',
      name: 'Knight',
      health: 100,
      maxHealth: 100,
      position: { x: 150, y: 150 },
      garrisonedIn: 'building-1',
    },
    'unit-2': {
      id: 'unit-2',
      name: 'Archer',
      health: 80,
      maxHealth: 80,
      position: { x: 200, y: 200 },
      garrisonedIn: 'building-1',
    },
    'unit-3': {
      id: 'unit-3',
      name: 'Pikeman',
      health: 90,
      maxHealth: 90,
      position: { x: 250, y: 250 },
      garrisonedIn: 'building-1',
    },
  },
  buildings: {
    'building-1': {
      id: 'building-1',
      name: 'Castle',
      position: { x: 300, y: 300 },
      health: 500,
      maxHealth: 500,
      garrisonedUnits: ['unit-1', 'unit-2', 'unit-3'],
      maxGarrison: 10,
    },
    'building-2': {
      id: 'building-2',
      name: 'Barracks',
      position: { x: 500, y: 300 },
      health: 300,
      maxHealth: 300,
      garrisonedUnits: [],
      maxGarrison: 5,
    },
  },
  selectedBuildingId: null,

  selectBuilding: (id) => set({ selectedBuildingId: id }),

  garrisonUnit: (unitId, buildingId) =>
    set((state) => {
      const unit = state.units[unitId];
      const building = state.buildings[buildingId];
      
      if (!unit || !building) return state;
      if (building.garrisonedUnits.length >= building.maxGarrison) return state;

      return {
        units: {
          ...state.units,
          [unitId]: { ...unit, garrisonedIn: buildingId },
        },
        buildings: {
          ...state.buildings,
          [buildingId]: {
            ...building,
            garrisonedUnits: [...building.garrisonedUnits, unitId],
          },
        },
      };
    }),

  releaseUnit: (unitId, buildingId) =>
    set((state) => {
      const unit = state.units[unitId];
      const building = state.buildings[buildingId];
      
      if (!unit || !building) return state;

      const exitPosition = {
        x: building.position.x - 50,
        y: building.position.y + 50,
      };

      return {
        units: {
          ...state.units,
          [unitId]: {
            ...unit,
            garrisonedIn: undefined,
            position: exitPosition,
          },
        },
        buildings: {
          ...state.buildings,
          [buildingId]: {
            ...building,
            garrisonedUnits: building.garrisonedUnits.filter((id) => id !== unitId),
          },
        },
      };
    }),

  // Release all units from a building
  releaseAllUnits: (buildingId) =>
    set((state) => {
      const building = state.buildings[buildingId];
      
      if (!building || building.garrisonedUnits.length === 0) return state;

      // Create updated units with cleared garrison status and exit positions
      const updatedUnits = { ...state.units };
      
      building.garrisonedUnits.forEach((unitId, index) => {
        const unit = state.units[unitId];
        if (unit) {
          // Calculate exit position (spread units around the building)
          const angle = (index / building.garrisonedUnits.length) * Math.PI * 2;
          const radius = 60;
          const exitPosition = {
            x: building.position.x + Math.cos(angle) * radius,
            y: building.position.y + Math.sin(angle) * radius,
          };

          updatedUnits[unitId] = {
            ...unit,
            garrisonedIn: undefined,
            position: exitPosition,
          };
        }
      });

      return {
        units: updatedUnits,
        buildings: {
          ...state.buildings,
          [buildingId]: {
            ...building,
            garrisonedUnits: [],
          },
        },
      };
    }),
}));
