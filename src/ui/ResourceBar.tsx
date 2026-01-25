import React from 'react';
import { useGameStore } from '../store/gameStore';

interface ResourceBarProps {
  className?: string;
}

export const ResourceBar: React.FC<ResourceBarProps> = ({ className = '' }) => {
  // Get resources and population data from the game store
  const resources = useGameStore((state) => state.resources);
  
  // FIXED: Use getPlayerPopulation() instead of worldPopulation
  // This shows ONLY the player's faction population, not the entire world
  const getPlayerPopulation = useGameStore((state) => state.getPlayerPopulation);
  const getPlayerMaxPopulation = useGameStore((state) => state.getPlayerMaxPopulation);
  
  const currentPopulation = getPlayerPopulation();
  const maxPopulation = getPlayerMaxPopulation();
  
  // Calculate population percentage for visual indicator
  const populationPercentage = maxPopulation > 0 
    ? (currentPopulation / maxPopulation) * 100 
    : 0;
  
  // Determine color based on population capacity
  const getPopulationColor = () => {
    if (populationPercentage >= 90) return 'text-red-500';
    if (populationPercentage >= 70) return 'text-yellow-500';
    return 'text-green-500';
  };
  
  return (
    <div className={`resource-bar flex items-center gap-6 bg-gray-800 px-4 py-2 rounded-lg shadow-lg ${className}`}>
      {/* Gold */}
      <div className="resource-item flex items-center gap-2">
        <span className="resource-icon text-yellow-400">💰</span>
        <span className="resource-label text-gray-300 text-sm">Gold:</span>
        <span className="resource-value text-white font-semibold">{resources.gold}</span>
      </div>
      
      {/* Wood */}
      <div className="resource-item flex items-center gap-2">
        <span className="resource-icon text-amber-600">🪵</span>
        <span className="resource-label text-gray-300 text-sm">Wood:</span>
        <span className="resource-value text-white font-semibold">{resources.wood}</span>
      </div>
      
      {/* Food */}
      <div className="resource-item flex items-center gap-2">
        <span className="resource-icon text-green-500">🌾</span>
        <span className="resource-label text-gray-300 text-sm">Food:</span>
        <span className="resource-value text-white font-semibold">{resources.food}</span>
      </div>
      
      {/* Population - FIXED: Now shows faction population only */}
      <div className="resource-item flex items-center gap-2">
        <span className="resource-icon text-blue-400">👥</span>
        <span className="resource-label text-gray-300 text-sm">Population:</span>
        <span className={`resource-value font-semibold ${getPopulationColor()}`}>
          {currentPopulation} / {maxPopulation}
        </span>
      </div>
      
      {/* Population bar indicator */}
      <div className="population-bar w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all duration-300 ${
            populationPercentage >= 90 ? 'bg-red-500' :
            populationPercentage >= 70 ? 'bg-yellow-500' : 
            'bg-green-500'
          }`}
          style={{ width: `${populationPercentage}%` }}
        />
      </div>
    </div>
  );
};

export default ResourceBar;
