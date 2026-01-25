import { create } from 'zustand';

export interface BuildingType {
  id: string;
  name: string;
  width: number;
  height: number;
  color: string;
}

export interface PlacedBuilding {
  id: string;
  type: BuildingType;
  x: number;
  y: number;
  rotation: number; // 0, 90, 180, 270 degrees
}

interface GameState {
  buildings: PlacedBuilding[];
  selectedBuildingType: BuildingType | null;
  ghostPosition: { x: number; y: number } | null;
  ghostRotation: number; // Current rotation of the ghost preview (0, 90, 180, 270)
  
  // Actions
  selectBuildingType: (buildingType: BuildingType | null) => void;
  setGhostPosition: (position: { x: number; y: number } | null) => void;
  rotateGhost: () => void;
  placeBuilding: () => void;
  removeBuilding: (id: string) => void;
  resetGhostRotation: () => void;
}

// Available building types
export const BUILDING_TYPES: BuildingType[] = [
  { id: 'barracks', name: 'Barracks', width: 3, height: 2, color: '#8B4513' },
  { id: 'tower', name: 'Tower', width: 2, height: 2, color: '#696969' },
  { id: 'farm', name: 'Farm', width: 4, height: 3, color: '#228B22' },
  { id: 'factory', name: 'Factory', width: 5, height: 4, color: '#4682B4' },
];

export const useGameStore = create<GameState>((set) => ({
  buildings: [],
  selectedBuildingType: null,
  ghostPosition: null,
  ghostRotation: 0,
  
  selectBuildingType: (buildingType) => 
    set({ 
      selectedBuildingType: buildingType,
      ghostRotation: 0 // Reset rotation when selecting a new building type
    }),
  
  setGhostPosition: (position) => 
    set({ ghostPosition: position }),
  
  rotateGhost: () => 
    set((state) => ({
      ghostRotation: (state.ghostRotation + 90) % 360
    })),
  
  resetGhostRotation: () =>
    set({ ghostRotation: 0 }),
  
  placeBuilding: () => 
    set((state) => {
      if (!state.selectedBuildingType || !state.ghostPosition) {
        return state;
      }
      
      const newBuilding: PlacedBuilding = {
        id: `building-${Date.now()}-${Math.random()}`,
        type: state.selectedBuildingType,
        x: state.ghostPosition.x,
        y: state.ghostPosition.y,
        rotation: state.ghostRotation, // Save the current rotation
      };
      
      return {
        buildings: [...state.buildings, newBuilding],
        ghostRotation: 0, // Reset rotation after placing
      };
    }),
  
  removeBuilding: (id) => 
    set((state) => ({
      buildings: state.buildings.filter((building) => building.id !== id)
    })),
}));
