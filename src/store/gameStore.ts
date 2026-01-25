import { create } from 'zustand';
import type { GameState, Faction, Unit, Resources } from '../types';

interface GameStore extends GameState {
  resources: Resources;
  
  // Population calculation methods
  getPlayerFaction: () => Faction | null;
  getPlayerPopulation: () => number;
  getPlayerMaxPopulation: () => number;
  getWorldPopulation: () => number;
  
  // Actions
  setPlayerFaction: (factionId: string) => void;
  addUnit: (unit: Unit) => void;
  removeUnit: (unitId: string, factionId: string) => void;
  updateResources: (resources: Partial<Resources>) => void;
}

export const useGameStore = create<GameStore>((set, get) => ({
  // Initial state
  factions: {},
  playerFactionId: null,
  worldPopulation: 0,
  resources: {
    gold: 1000,
    wood: 500,
    food: 500,
  },
  
  // Get the player's faction
  getPlayerFaction: () => {
    const { factions, playerFactionId } = get();
    if (!playerFactionId) return null;
    return factions[playerFactionId] || null;
  },
  
  // Get ONLY the player faction's population (not world population)
  getPlayerPopulation: () => {
    const playerFaction = get().getPlayerFaction();
    if (!playerFaction) return 0;
    
    // Calculate population from the number of units in player's faction
    return playerFaction.units.length;
  },
  
  // Get the player faction's max population
  getPlayerMaxPopulation: () => {
    const playerFaction = get().getPlayerFaction();
    if (!playerFaction) return 0;
    return playerFaction.maxPopulation;
  },
  
  // Get total world population (all factions combined)
  getWorldPopulation: () => {
    const { factions } = get();
    return Object.values(factions).reduce(
      (total, faction) => total + faction.units.length,
      0
    );
  },
  
  // Set the player's faction
  setPlayerFaction: (factionId: string) => {
    set({ playerFactionId: factionId });
  },
  
  // Add a unit to a faction
  addUnit: (unit: Unit) => {
    set((state) => {
      const faction = state.factions[unit.factionId];
      if (!faction) return state;
      
      const updatedFaction = {
        ...faction,
        units: [...faction.units, unit],
        population: faction.units.length + 1,
      };
      
      return {
        factions: {
          ...state.factions,
          [unit.factionId]: updatedFaction,
        },
        worldPopulation: state.worldPopulation + 1,
      };
    });
  },
  
  // Remove a unit from a faction
  removeUnit: (unitId: string, factionId: string) => {
    set((state) => {
      const faction = state.factions[factionId];
      if (!faction) return state;
      
      const updatedFaction = {
        ...faction,
        units: faction.units.filter((u) => u.id !== unitId),
        population: faction.units.length - 1,
      };
      
      return {
        factions: {
          ...state.factions,
          [factionId]: updatedFaction,
        },
        worldPopulation: state.worldPopulation - 1,
      };
    });
  },
  
  // Update resources
  updateResources: (newResources: Partial<Resources>) => {
    set((state) => ({
      resources: {
        ...state.resources,
        ...newResources,
      },
    }));
  },
}));
