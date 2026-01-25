import { create } from 'zustand';

export interface Unit {
  id: string;
  position: { x: number; y: number; z: number };
  destination?: { x: number; y: number; z: number };
  path?: Array<{ x: number; y: number; z: number }>;
  isSelected: boolean;
  isLeadUnit?: boolean;
}

export type PathVisibilityMode = 
  | 'all'           // Show all unit paths
  | 'lead-only'     // Show only lead unit path
  | 'group-marker'  // Show single group destination marker
  | 'none'          // Hide all paths
  | 'fade-quick';   // Show paths with quick fade

interface GameState {
  units: Unit[];
  pathVisibilityMode: PathVisibilityMode;
  showPathLines: boolean;
  pathFadeDuration: number; // in milliseconds
  pathOpacity: number;
  groupDestinationMarkerEnabled: boolean;
  
  // Actions
  setPathVisibilityMode: (mode: PathVisibilityMode) => void;
  setShowPathLines: (show: boolean) => void;
  setPathFadeDuration: (duration: number) => void;
  setPathOpacity: (opacity: number) => void;
  setGroupDestinationMarkerEnabled: (enabled: boolean) => void;
  addUnit: (unit: Unit) => void;
  updateUnit: (id: string, updates: Partial<Unit>) => void;
  removeUnit: (id: string) => void;
  selectUnits: (ids: string[]) => void;
  deselectAllUnits: () => void;
  setUnitDestination: (id: string, destination: { x: number; y: number; z: number }) => void;
  calculateGroupDestination: () => { x: number; y: number; z: number } | null;
}

export const useGameStore = create<GameState>((set, get) => ({
  units: [],
  pathVisibilityMode: 'lead-only', // Default to showing only lead unit
  showPathLines: true,
  pathFadeDuration: 1000, // 1 second fade
  pathOpacity: 0.7,
  groupDestinationMarkerEnabled: true,
  
  setPathVisibilityMode: (mode) => set({ pathVisibilityMode: mode }),
  
  setShowPathLines: (show) => set({ showPathLines: show }),
  
  setPathFadeDuration: (duration) => set({ pathFadeDuration: duration }),
  
  setPathOpacity: (opacity) => set({ pathOpacity: Math.max(0, Math.min(1, opacity)) }),
  
  setGroupDestinationMarkerEnabled: (enabled) => set({ groupDestinationMarkerEnabled: enabled }),
  
  addUnit: (unit) => set((state) => ({
    units: [...state.units, unit]
  })),
  
  updateUnit: (id, updates) => set((state) => ({
    units: state.units.map(unit => 
      unit.id === id ? { ...unit, ...updates } : unit
    )
  })),
  
  removeUnit: (id) => set((state) => ({
    units: state.units.filter(unit => unit.id !== id)
  })),
  
  selectUnits: (ids) => set((state) => {
    const selectedUnits = state.units
      .filter(unit => ids.includes(unit.id))
      .map((unit, index) => ({
        ...unit,
        isSelected: true,
        isLeadUnit: index === 0 // First selected unit is the lead
      }));
    
    const unselectedUnits = state.units
      .filter(unit => !ids.includes(unit.id))
      .map(unit => ({
        ...unit,
        isSelected: false,
        isLeadUnit: false
      }));
    
    return { units: [...selectedUnits, ...unselectedUnits] };
  }),
  
  deselectAllUnits: () => set((state) => ({
    units: state.units.map(unit => ({
      ...unit,
      isSelected: false,
      isLeadUnit: false
    }))
  })),
  
  setUnitDestination: (id, destination) => set((state) => ({
    units: state.units.map(unit =>
      unit.id === id ? { ...unit, destination, path: [unit.position, destination] } : unit
    )
  })),
  
  calculateGroupDestination: () => {
    const selectedUnits = get().units.filter(unit => unit.isSelected && unit.destination);
    
    if (selectedUnits.length === 0) return null;
    
    // Calculate centroid of all destinations
    const sum = selectedUnits.reduce(
      (acc, unit) => ({
        x: acc.x + (unit.destination?.x || 0),
        y: acc.y + (unit.destination?.y || 0),
        z: acc.z + (unit.destination?.z || 0)
      }),
      { x: 0, y: 0, z: 0 }
    );
    
    return {
      x: sum.x / selectedUnits.length,
      y: sum.y / selectedUnits.length,
      z: sum.z / selectedUnits.length
    };
  }
}));
