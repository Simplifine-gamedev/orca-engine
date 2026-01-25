import React from 'react';
import { useGameStore } from '../store/gameStore';

export const ResourceBar: React.FC = () => {
  // Fixed: Use getPlayerPopulation instead of getWorldPopulation
  const playerPopulation = useGameStore(state => state.getPlayerPopulation());
  const maxPopulation = useGameStore(state => state.getPlayerMaxPopulation());
  const playerFaction = useGameStore(state => state.getPlayerFaction());
  
  return (
    <div className="resource-bar">
      <div className="population-counter">
        <span className="population-icon">👥</span>
        <span className="population-text">
          {playerPopulation} / {maxPopulation}
        </span>
      </div>
      {playerFaction && (
        <div className="faction-info">
          <span 
            className="faction-color" 
            style={{ backgroundColor: playerFaction.color }}
          />
          <span className="faction-name">{playerFaction.name}</span>
        </div>
      )}
    </div>
  );
};
