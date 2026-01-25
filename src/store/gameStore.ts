import { create } from 'zustand';

export interface Unit {
  id: string;
  type: string;
  factionId: string;
  health: number;
  maxHealth: number;
}

export interface Faction {
  id: string;
  name: string;
  color: string;
  isPlayer: boolean;
}

export interface GameState {
  units: Unit[];
  factions: Faction[];
  playerFactionId: string | null;
  
  // Selectors
  getPlayerFaction: () => Faction | null;
  getPlayerUnits: () => Unit[];
  getPlayerPopulation: () => number;
  getPlayerMaxPopulation: () => number;
  getWorldPopulation: () => number;
  
  // Actions
  addUnit: (unit: Unit) => void;
  removeUnit: (unitId: string) => void;
  setPlayerFaction: (factionId: string) => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  units: [],
  factions: [],
  playerFactionId: null,
  
  getPlayerFaction: () => {
    const state = get();
    if (!state.playerFactionId) return null;
    return state.factions.find(f => f.id === state.playerFactionId) || null;
  },
  
  getPlayerUnits: () => {
    const state = get();
    if (!state.playerFactionId) return [];
    return state.units.filter(unit => unit.factionId === state.playerFactionId);
  },
  
  getPlayerPopulation: () => {
    const state = get();
    if (!state.playerFactionId) return 0;
    return state.units.filter(unit => unit.factionId === state.playerFactionId).length;
  },
  
  getPlayerMaxPopulation: () => {
    // TODO: Calculate based on buildings/upgrades
    // For now, return a fixed value
    return 100;
  },
  
  getWorldPopulation: () => {
    return get().units.length;
  },
  
  addUnit: (unit: Unit) => {
    set(state => ({
      units: [...state.units, unit]
    }));
  },
  
  removeUnit: (unitId: string) => {
    set(state => ({
      units: state.units.filter(u => u.id !== unitId)
    }));
  },
  
  setPlayerFaction: (factionId: string) => {
    set({ playerFactionId: factionId });
  }
}));
