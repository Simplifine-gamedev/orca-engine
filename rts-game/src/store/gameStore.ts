import { create } from 'zustand'
import { Unit, Position, FormationType, SpreadType, GameState } from '../types'
import { calculateFormationPositions, calculateCenterPosition, calculateAngle } from '../utils/formations'

interface GameStore extends GameState {
  // Actions
  addUnit: (position: Position) => void
  selectUnits: (unitIds: string[]) => void
  toggleUnitSelection: (unitId: string) => void
  clearSelection: () => void
  moveSelectedUnits: (targetPosition: Position) => void
  startFormationDrag: (position: Position) => void
  updateFormationDrag: (position: Position) => void
  endFormationDrag: () => void
  setFormationType: (type: FormationType) => void
  setSpreadType: (spread: SpreadType) => void
  toggleIndividualPaths: () => void
  updateUnits: () => void
}

const UNIT_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']

export const useGameStore = create<GameStore>((set, get) => ({
  // Initial state
  units: [],
  selectedUnitIds: [],
  isDraggingFormation: false,
  formationDragStart: null,
  formationDragEnd: null,
  formationConfig: {
    type: 'line',
    spread: 'normal',
    facing: 0,
    showIndividualPaths: false,
  },

  // Actions
  addUnit: (position: Position) => {
    const newUnit: Unit = {
      id: `unit-${Date.now()}-${Math.random()}`,
      position,
      targetPosition: null,
      isSelected: false,
      color: UNIT_COLORS[get().units.length % UNIT_COLORS.length],
      facing: 0,
    }
    
    set((state) => ({
      units: [...state.units, newUnit],
    }))
  },

  selectUnits: (unitIds: string[]) => {
    set((state) => ({
      selectedUnitIds: unitIds,
      units: state.units.map((unit) => ({
        ...unit,
        isSelected: unitIds.includes(unit.id),
      })),
    }))
  },

  toggleUnitSelection: (unitId: string) => {
    set((state) => {
      const isCurrentlySelected = state.selectedUnitIds.includes(unitId)
      const newSelectedIds = isCurrentlySelected
        ? state.selectedUnitIds.filter((id) => id !== unitId)
        : [...state.selectedUnitIds, unitId]
      
      return {
        selectedUnitIds: newSelectedIds,
        units: state.units.map((unit) => ({
          ...unit,
          isSelected: newSelectedIds.includes(unit.id),
        })),
      }
    })
  },

  clearSelection: () => {
    set((state) => ({
      selectedUnitIds: [],
      units: state.units.map((unit) => ({
        ...unit,
        isSelected: false,
      })),
    }))
  },

  moveSelectedUnits: (targetPosition: Position) => {
    const { selectedUnitIds, units, formationConfig } = get()
    
    if (selectedUnitIds.length === 0) return
    
    const selectedUnits = units.filter((u) => selectedUnitIds.includes(u.id))
    const currentCenter = calculateCenterPosition(selectedUnits.map((u) => u.position))
    
    // Calculate formation positions
    const formationPositions = calculateFormationPositions(
      targetPosition,
      selectedUnits.length,
      formationConfig.type,
      formationConfig.spread,
      formationConfig.facing
    )
    
    // Assign formation positions to units (maintain relative order)
    set((state) => ({
      units: state.units.map((unit) => {
        const selectedIndex = selectedUnitIds.indexOf(unit.id)
        if (selectedIndex !== -1 && formationPositions[selectedIndex]) {
          return {
            ...unit,
            targetPosition: formationPositions[selectedIndex],
            facing: formationConfig.facing,
          }
        }
        return unit
      }),
    }))
  },

  startFormationDrag: (position: Position) => {
    const { selectedUnitIds, units } = get()
    
    if (selectedUnitIds.length === 0) return
    
    const selectedUnits = units.filter((u) => selectedUnitIds.includes(u.id))
    const center = calculateCenterPosition(selectedUnits.map((u) => u.position))
    
    set({
      isDraggingFormation: true,
      formationDragStart: center,
      formationDragEnd: position,
    })
  },

  updateFormationDrag: (position: Position) => {
    const { isDraggingFormation, formationDragStart } = get()
    
    if (!isDraggingFormation || !formationDragStart) return
    
    const angle = calculateAngle(formationDragStart, position)
    
    set((state) => ({
      formationDragEnd: position,
      formationConfig: {
        ...state.formationConfig,
        facing: angle,
      },
    }))
  },

  endFormationDrag: () => {
    const { formationDragStart, formationDragEnd, isDraggingFormation } = get()
    
    if (isDraggingFormation && formationDragStart && formationDragEnd) {
      // Move units to the drag start position with the calculated facing
      get().moveSelectedUnits(formationDragStart)
    }
    
    set({
      isDraggingFormation: false,
      formationDragStart: null,
      formationDragEnd: null,
    })
  },

  setFormationType: (type: FormationType) => {
    set((state) => ({
      formationConfig: {
        ...state.formationConfig,
        type,
      },
    }))
  },

  setSpreadType: (spread: SpreadType) => {
    set((state) => ({
      formationConfig: {
        ...state.formationConfig,
        spread,
      },
    }))
  },

  toggleIndividualPaths: () => {
    set((state) => ({
      formationConfig: {
        ...state.formationConfig,
        showIndividualPaths: !state.formationConfig.showIndividualPaths,
      },
    }))
  },

  updateUnits: () => {
    set((state) => ({
      units: state.units.map((unit) => {
        if (!unit.targetPosition) return unit
        
        const dx = unit.targetPosition.x - unit.position.x
        const dy = unit.targetPosition.y - unit.position.y
        const distance = Math.sqrt(dx * dx + dy * dy)
        
        if (distance < 2) {
          // Reached target
          return {
            ...unit,
            position: unit.targetPosition,
            targetPosition: null,
          }
        }
        
        // Move towards target
        const speed = 2
        const moveX = (dx / distance) * speed
        const moveY = (dy / distance) * speed
        
        return {
          ...unit,
          position: {
            x: unit.position.x + moveX,
            y: unit.position.y + moveY,
          },
        }
      }),
    }))
  },
}))
