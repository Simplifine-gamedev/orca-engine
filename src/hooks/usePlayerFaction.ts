import { useGameStore } from '../store/gameStore';
import type { Faction } from '../types';

/**
 * Custom hook to get player faction data
 * This ensures we always work with the player's faction, not world data
 */
export const usePlayerFaction = () => {
  const getPlayerFaction = useGameStore((state) => state.getPlayerFaction);
  const getPlayerPopulation = useGameStore((state) => state.getPlayerPopulation);
  const getPlayerMaxPopulation = useGameStore((state) => state.getPlayerMaxPopulation);
  
  const playerFaction = getPlayerFaction();
  const population = getPlayerPopulation();
  const maxPopulation = getPlayerMaxPopulation();
  
  return {
    faction: playerFaction,
    population,
    maxPopulation,
    isPopulationFull: population >= maxPopulation,
    populationPercentage: maxPopulation > 0 ? (population / maxPopulation) * 100 : 0,
  };
};
