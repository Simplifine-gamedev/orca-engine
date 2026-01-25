import { create } from 'zustand';
import { Unit, Position, ControlGroup, ActionType } from '../types';

interface GameState {
  units: Unit[];
  selectedUnitIds: string[];
  controlGroups: ControlGroup;
  currentActionType: ActionType;
  hoveredUnitId: string | null;
  
  // Selection actions
  selectUnit: (unitId: string, addToSelection: boolean) => void;
  selectUnits: (unitIds: string[]) => void;
  deselectAll: () => void;
  
  // Control groups
  saveControlGroup: (groupNumber: number) => void;
  recallControlGroup: (groupNumber: number) => void;
  
  // Unit cycling
  cycleSelectedUnits: () => void;
  
  // Unit actions
  moveUnitsTo: (position: Position) => void;
  attackUnit: (targetId: string) => void;
  
  // Hover state
  setHoveredUnit: (unitId: string | null) => void;
  
  // Game loop
  updateUnits: () => void;
  
  // Init
  initializeUnits: () => void;
}

const createUnit = (
  id: string,
  x: number,
  y: number,
  team: 'player' | 'enemy',
  type: 'soldier' | 'tank' | 'scout'
): Unit => ({
  id,
  position: { x, y },
  targetPosition: null,
  health: 100,
  maxHealth: 100,
  team,
  isSelected: false,
  type,
  speed: type === 'scout' ? 3 : type === 'soldier' ? 2 : 1.5,
});

export const useGameStore = create<GameState>((set, get) => ({
  units: [],
  selectedUnitIds: [],
  controlGroups: {},
  currentActionType: null,
  hoveredUnitId: null,

  initializeUnits: () => {
    const units: Unit[] = [
      // Player units
      createUnit('player-1', 200, 300, 'player', 'soldier'),
      createUnit('player-2', 250, 300, 'player', 'soldier'),
      createUnit('player-3', 300, 300, 'player', 'tank'),
      createUnit('player-4', 350, 300, 'player', 'scout'),
      createUnit('player-5', 200, 350, 'player', 'soldier'),
      createUnit('player-6', 250, 350, 'player', 'tank'),
      
      // Enemy units
      createUnit('enemy-1', 600, 200, 'enemy', 'soldier'),
      createUnit('enemy-2', 650, 200, 'enemy', 'soldier'),
      createUnit('enemy-3', 700, 200, 'enemy', 'tank'),
    ];
    
    set({ units });
  },

  selectUnit: (unitId: string, addToSelection: boolean) => {
    const { units, selectedUnitIds } = get();
    const unit = units.find(u => u.id === unitId);
    
    if (!unit || unit.team !== 'player') return;

    let newSelectedIds: string[];

    if (addToSelection) {
      // Toggle selection
      if (selectedUnitIds.includes(unitId)) {
        newSelectedIds = selectedUnitIds.filter(id => id !== unitId);
      } else {
        newSelectedIds = [...selectedUnitIds, unitId];
      }
    } else {
      newSelectedIds = [unitId];
    }

    set({
      selectedUnitIds: newSelectedIds,
      units: units.map(u => ({
        ...u,
        isSelected: newSelectedIds.includes(u.id),
      })),
    });
  },

  selectUnits: (unitIds: string[]) => {
    const { units } = get();
    const playerUnitIds = unitIds.filter(id => {
      const unit = units.find(u => u.id === id);
      return unit && unit.team === 'player';
    });

    set({
      selectedUnitIds: playerUnitIds,
      units: units.map(u => ({
        ...u,
        isSelected: playerUnitIds.includes(u.id),
      })),
    });
  },

  deselectAll: () => {
    const { units } = get();
    set({
      selectedUnitIds: [],
      units: units.map(u => ({ ...u, isSelected: false })),
    });
  },

  saveControlGroup: (groupNumber: number) => {
    const { selectedUnitIds, controlGroups } = get();
    if (selectedUnitIds.length === 0) return;

    set({
      controlGroups: {
        ...controlGroups,
        [groupNumber]: [...selectedUnitIds],
      },
    });
  },

  recallControlGroup: (groupNumber: number) => {
    const { controlGroups } = get();
    const unitIds = controlGroups[groupNumber];
    
    if (unitIds && unitIds.length > 0) {
      get().selectUnits(unitIds);
    }
  },

  cycleSelectedUnits: () => {
    const { selectedUnitIds } = get();
    if (selectedUnitIds.length <= 1) return;

    // Move first selected unit to end
    const [first, ...rest] = selectedUnitIds;
    const newSelectedIds = [...rest, first];

    set({ selectedUnitIds: newSelectedIds });
  },

  moveUnitsTo: (position: Position) => {
    const { units, selectedUnitIds } = get();
    
    set({
      units: units.map(u => {
        if (selectedUnitIds.includes(u.id)) {
          // Add some spacing for multiple units
          const index = selectedUnitIds.indexOf(u.id);
          const offset = index * 40;
          return {
            ...u,
            targetPosition: {
              x: position.x + (offset % 80) - 40,
              y: position.y + Math.floor(offset / 80) * 40 - 40,
            },
          };
        }
        return u;
      }),
      currentActionType: 'move',
    });

    // Clear action type after a short delay
    setTimeout(() => {
      set({ currentActionType: null });
    }, 500);
  },

  attackUnit: (targetId: string) => {
    const { units, selectedUnitIds } = get();
    const target = units.find(u => u.id === targetId);
    
    if (!target) return;

    // Move selected units towards target
    set({
      units: units.map(u => {
        if (selectedUnitIds.includes(u.id)) {
          return {
            ...u,
            targetPosition: { ...target.position },
          };
        }
        if (u.id === targetId) {
          // Damage the target
          return {
            ...u,
            health: Math.max(0, u.health - 20),
          };
        }
        return u;
      }),
      currentActionType: 'attack',
    });

    // Clear action type after a short delay
    setTimeout(() => {
      set({ currentActionType: null });
    }, 500);
  },

  setHoveredUnit: (unitId: string | null) => {
    set({ hoveredUnitId: unitId });
  },

  updateUnits: () => {
    const { units } = get();
    
    set({
      units: units.map(u => {
        if (u.targetPosition) {
          const dx = u.targetPosition.x - u.position.x;
          const dy = u.targetPosition.y - u.position.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < u.speed) {
            // Reached target
            return {
              ...u,
              position: u.targetPosition,
              targetPosition: null,
            };
          } else {
            // Move towards target
            const ratio = u.speed / distance;
            return {
              ...u,
              position: {
                x: u.position.x + dx * ratio,
                y: u.position.y + dy * ratio,
              },
            };
          }
        }
        return u;
      }),
    });
  },
}));
