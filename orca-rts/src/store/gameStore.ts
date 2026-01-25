import { create } from 'zustand'

export type BuildingType = 'barracks' | 'factory' | 'powerPlant' | 'mine'

export interface BuildingData {
  id: string
  type: BuildingType
  x: number
  y: number
  rotation: number // 0, 90, 180, 270 degrees
  width: number
  height: number
}

export interface GameState {
  buildings: BuildingData[]
  selectedBuildingType: BuildingType | null
  ghostBuilding: {
    x: number
    y: number
    rotation: number
    type: BuildingType | null
  } | null
  isPlacingBuilding: boolean
  
  // Actions
  setSelectedBuildingType: (type: BuildingType | null) => void
  updateGhostBuilding: (x: number, y: number) => void
  rotateGhostBuilding: () => void
  placeBuilding: () => void
  cancelPlacement: () => void
}

export const useGameStore = create<GameState>((set) => ({
  buildings: [],
  selectedBuildingType: null,
  ghostBuilding: null,
  isPlacingBuilding: false,
  
  setSelectedBuildingType: (type) => set({
    selectedBuildingType: type,
    isPlacingBuilding: type !== null,
    ghostBuilding: type ? {
      x: 0,
      y: 0,
      rotation: 0,
      type
    } : null
  }),
  
  updateGhostBuilding: (x, y) => set((state) => {
    if (!state.ghostBuilding) return {}
    return {
      ghostBuilding: {
        ...state.ghostBuilding,
        x,
        y
      }
    }
  }),
  
  rotateGhostBuilding: () => set((state) => {
    if (!state.ghostBuilding) return {}
    const newRotation = (state.ghostBuilding.rotation + 90) % 360
    return {
      ghostBuilding: {
        ...state.ghostBuilding,
        rotation: newRotation
      }
    }
  }),
  
  placeBuilding: () => set((state) => {
    if (!state.ghostBuilding || !state.ghostBuilding.type) return {}
    
    const buildingSize = getBuildingSize(state.ghostBuilding.type)
    const newBuilding: BuildingData = {
      id: `building-${Date.now()}`,
      type: state.ghostBuilding.type,
      x: state.ghostBuilding.x,
      y: state.ghostBuilding.y,
      rotation: state.ghostBuilding.rotation,
      width: buildingSize.width,
      height: buildingSize.height
    }
    
    return {
      buildings: [...state.buildings, newBuilding],
      selectedBuildingType: state.selectedBuildingType, // Keep selection active
      ghostBuilding: {
        ...state.ghostBuilding,
        rotation: state.ghostBuilding.rotation // Keep current rotation
      }
    }
  }),
  
  cancelPlacement: () => set({
    selectedBuildingType: null,
    ghostBuilding: null,
    isPlacingBuilding: false
  })
}))

export function getBuildingSize(type: BuildingType): { width: number; height: number } {
  switch (type) {
    case 'barracks':
      return { width: 80, height: 80 }
    case 'factory':
      return { width: 120, height: 100 }
    case 'powerPlant':
      return { width: 100, height: 100 }
    case 'mine':
      return { width: 60, height: 60 }
    default:
      return { width: 80, height: 80 }
  }
}

export function getBuildingColor(type: BuildingType): string {
  switch (type) {
    case 'barracks':
      return '#e74c3c'
    case 'factory':
      return '#3498db'
    case 'powerPlant':
      return '#f39c12'
    case 'mine':
      return '#9b59b6'
    default:
      return '#95a5a6'
  }
}
