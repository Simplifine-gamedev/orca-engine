import React from 'react';
import { useGameStore } from '../store/gameStore';

export const ResourceBar: React.FC = () => {
  const resources = useGameStore((state) => state.resources);

  return (
    <div className="resource-bar">
      <div className="resource-item">
        <span className="resource-icon">🪵</span>
        <span className="resource-value">{resources.wood}</span>
      </div>
      <div className="resource-item">
        <span className="resource-icon">🪙</span>
        <span className="resource-value">{resources.gold}</span>
      </div>
      <div className="resource-item">
        <span className="resource-icon">🪨</span>
        <span className="resource-value">{resources.stone}</span>
      </div>
    </div>
  );
};
