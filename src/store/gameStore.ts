import { create } from 'zustand';
import { Unit, SelectionBox, ControlGroups, Position } from '../types/unit';

interface GameState {
  // Units
  units: Unit[];
  selectedUnitIds: string[];
  
  // Selection
  selectionBox: SelectionBox | null;
  isSelecting: boolean;
  
  // Control groups
  controlGroups: ControlGroups;
  
  // Movement
  showMoveIndicator: boolean;
  moveIndicatorPosition: Position | null;
  
  // Actions
  addUnit: (unit: Unit) => void;
  selectUnits: (unitIds: string[]) => void;
  addToSelection: (unitIds: string[]) => void;
  removeFromSelection: (unitIds: string[]) => void;
  clearSelection: () => void;
  
  // Selection box
  startSelection: (x: number, y: number) => void;
  updateSelection: (x: number, y: number) => void;
  endSelection: () => void;
  
  // Control groups
  assignControlGroup: (groupNumber: number, unitIds: string[]) => void;
  selectControlGroup: (groupNumber: number, additive?: boolean) => void;
  
  // Cycle selection
  cycleSelectedUnits: () => void;
  
  // Movement
  moveUnits: (unitIds: string[], targetPosition: Position) => void;
  updateUnitPositions: () => void;
  
  // Move indicator
  showMoveIndicatorAt: (position: Position) => void;
  hideMoveIndicator: () => void;
  
  // Get selected units
  getSelectedUnits: () => Unit[];
}

export const useGameStore = create<GameState>((set, get) => ({
  // Initial state
  units: [],
  selectedUnitIds: [],
  selectionBox: null,
  isSelecting: false,
  controlGroups: {},
  showMoveIndicator: false,
  moveIndicatorPosition: null,
  
  // Add unit
  addUnit: (unit) => set((state) => ({
    units: [...state.units, unit]
  })),
  
  // Selection methods
  selectUnits: (unitIds) => set({
    selectedUnitIds: unitIds
  }),
  
  addToSelection: (unitIds) => set((state) => ({
    selectedUnitIds: [...new Set([...state.selectedUnitIds, ...unitIds])]
  })),
  
  removeFromSelection: (unitIds) => set((state) => ({
    selectedUnitIds: state.selectedUnitIds.filter(id => !unitIds.includes(id))
  })),
  
  clearSelection: () => set({ selectedUnitIds: [] }),
  
  // Selection box methods
  startSelection: (x, y) => set({
    selectionBox: { startX: x, startY: y, endX: x, endY: y },
    isSelecting: true
  }),
  
  updateSelection: (x, y) => set((state) => {
    if (!state.selectionBox) return state;
    return {
      selectionBox: {
        ...state.selectionBox,
        endX: x,
        endY: y
      }
    };
  }),
  
  endSelection: () => {
    const state = get();
    if (!state.selectionBox) return;
    
    const { startX, startY, endX, endY } = state.selectionBox;
    const minX = Math.min(startX, endX);
    const maxX = Math.max(startX, endX);
    const minY = Math.min(startY, endY);
    const maxY = Math.max(startY, endY);
    
    const selectedUnits = state.units.filter(unit => 
      unit.team === 'player' &&
      unit.position.x >= minX &&
      unit.position.x <= maxX &&
      unit.position.y >= minY &&
      unit.position.y <= maxY
    );
    
    set({
      selectedUnitIds: selectedUnits.map(u => u.id),
      selectionBox: null,
      isSelecting: false
    });
  },
  
  // Control group methods
  assignControlGroup: (groupNumber, unitIds) => set((state) => ({
    controlGroups: {
      ...state.controlGroups,
      [groupNumber]: unitIds
    }
  })),
  
  selectControlGroup: (groupNumber, additive = false) => {
    const state = get();
    const groupUnitIds = state.controlGroups[groupNumber] || [];
    
    if (additive) {
      state.addToSelection(groupUnitIds);
    } else {
      state.selectUnits(groupUnitIds);
    }
  },
  
  // Cycle through selected units
  cycleSelectedUnits: () => {
    const state = get();
    if (state.selectedUnitIds.length <= 1) return;
    
    const [first, ...rest] = state.selectedUnitIds;
    set({
      selectedUnitIds: [...rest, first]
    });
  },
  
  // Movement methods
  moveUnits: (unitIds, targetPosition) => set((state) => ({
    units: state.units.map(unit => 
      unitIds.includes(unit.id)
        ? { ...unit, targetPosition, isMoving: true }
        : unit
    )
  })),
  
  updateUnitPositions: () => set((state) => ({
    units: state.units.map(unit => {
      if (!unit.isMoving || !unit.targetPosition) return unit;
      
      const dx = unit.targetPosition.x - unit.position.x;
      const dy = unit.targetPosition.y - unit.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < unit.speed) {
        return {
          ...unit,
          position: unit.targetPosition,
          targetPosition: null,
          isMoving: false
        };
      }
      
      const ratio = unit.speed / distance;
      return {
        ...unit,
        position: {
          x: unit.position.x + dx * ratio,
          y: unit.position.y + dy * ratio
        }
      };
    })
  })),
  
  // Move indicator methods
  showMoveIndicatorAt: (position) => set({
    showMoveIndicator: true,
    moveIndicatorPosition: position
  }),
  
  hideMoveIndicator: () => set({
    showMoveIndicator: false,
    moveIndicatorPosition: null
  }),
  
  // Get selected units
  getSelectedUnits: () => {
    const state = get();
    return state.units.filter(u => state.selectedUnitIds.includes(u.id));
  }
}));
