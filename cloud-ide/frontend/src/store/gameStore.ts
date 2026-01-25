import { create } from 'zustand'

export interface Unit {
  id: string
  x: number
  y: number
  targetX: number | null
  targetY: number | null
  isSelected: boolean
  isLeader: boolean
}

export interface PathSettings {
  showPaths: boolean
  showOnlyLeadUnit: boolean
  showGroupDestination: boolean
  pathFadeSpeed: number
  pathOpacity: number
}

interface GameState {
  units: Unit[]
  selectedUnits: string[]
  pathSettings: PathSettings
  
  // Unit actions
  addUnit: (x: number, y: number) => void
  selectUnit: (id: string, multiSelect?: boolean) => void
  selectUnits: (ids: string[]) => void
  moveUnits: (targetX: number, targetY: number) => void
  clearSelection: () => void
  
  // Path visibility settings
  togglePathVisibility: () => void
  toggleLeadUnitOnly: () => void
  toggleGroupDestination: () => void
  setPathOpacity: (opacity: number) => void
  setPathFadeSpeed: (speed: number) => void
}

const useGameStore = create<GameState>((set, get) => ({
  units: [],
  selectedUnits: [],
  pathSettings: {
    showPaths: true,
    showOnlyLeadUnit: false,
    showGroupDestination: true,
    pathFadeSpeed: 1.0,
    pathOpacity: 0.8,
  },
  
  addUnit: (x: number, y: number) => {
    const newUnit: Unit = {
      id: `unit-${Date.now()}-${Math.random()}`,
      x,
      y,
      targetX: null,
      targetY: null,
      isSelected: false,
      isLeader: false,
    }
    set((state) => ({
      units: [...state.units, newUnit],
    }))
  },
  
  selectUnit: (id: string, multiSelect = false) => {
    set((state) => {
      const newSelectedUnits = multiSelect
        ? state.selectedUnits.includes(id)
          ? state.selectedUnits.filter((uid) => uid !== id)
          : [...state.selectedUnits, id]
        : [id]
      
      // Update units with selection state and leader designation
      const units = state.units.map((unit, index) => ({
        ...unit,
        isSelected: newSelectedUnits.includes(unit.id),
        isLeader: newSelectedUnits[0] === unit.id,
      }))
      
      return { selectedUnits: newSelectedUnits, units }
    })
  },
  
  selectUnits: (ids: string[]) => {
    set((state) => {
      const units = state.units.map((unit, index) => ({
        ...unit,
        isSelected: ids.includes(unit.id),
        isLeader: ids[0] === unit.id,
      }))
      
      return { selectedUnits: ids, units }
    })
  },
  
  moveUnits: (targetX: number, targetY: number) => {
    set((state) => {
      const units = state.units.map((unit) => {
        if (state.selectedUnits.includes(unit.id)) {
          return {
            ...unit,
            targetX,
            targetY,
          }
        }
        return unit
      })
      
      return { units }
    })
  },
  
  clearSelection: () => {
    set((state) => ({
      selectedUnits: [],
      units: state.units.map((unit) => ({
        ...unit,
        isSelected: false,
        isLeader: false,
      })),
    }))
  },
  
  togglePathVisibility: () => {
    set((state) => ({
      pathSettings: {
        ...state.pathSettings,
        showPaths: !state.pathSettings.showPaths,
      },
    }))
  },
  
  toggleLeadUnitOnly: () => {
    set((state) => ({
      pathSettings: {
        ...state.pathSettings,
        showOnlyLeadUnit: !state.pathSettings.showOnlyLeadUnit,
      },
    }))
  },
  
  toggleGroupDestination: () => {
    set((state) => ({
      pathSettings: {
        ...state.pathSettings,
        showGroupDestination: !state.pathSettings.showGroupDestination,
      },
    }))
  },
  
  setPathOpacity: (opacity: number) => {
    set((state) => ({
      pathSettings: {
        ...state.pathSettings,
        pathOpacity: Math.max(0, Math.min(1, opacity)),
      },
    }))
  },
  
  setPathFadeSpeed: (speed: number) => {
    set((state) => ({
      pathSettings: {
        ...state.pathSettings,
        pathFadeSpeed: Math.max(0.1, Math.min(5, speed)),
      },
    }))
  },
}))

export default useGameStore
