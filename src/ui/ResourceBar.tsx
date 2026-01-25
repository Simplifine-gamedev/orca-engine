import React from 'react';
import { useGameStore } from '../store/gameStore';

interface ResourceBarProps {
  className?: string;
}

const ResourceBar: React.FC<ResourceBarProps> = ({ className = '' }) => {
  const resources = useGameStore((state) => state.resources);
  const isPaused = useGameStore((state) => state.isPaused);
  const gameSpeed = useGameStore((state) => state.gameSpeed);
  const togglePause = useGameStore((state) => state.togglePause);
  const setGameSpeed = useGameStore((state) => state.setGameSpeed);

  const resourceItems = [
    { name: 'Wood', value: resources.wood, icon: '🪵', color: '#8B4513' },
    { name: 'Gold', value: resources.gold, icon: '💰', color: '#FFD700' },
    { name: 'Stone', value: resources.stone, icon: '🪨', color: '#808080' },
    { name: 'Food', value: resources.food, icon: '🌾', color: '#F4A460' },
  ];

  return (
    <div 
      className={`resource-bar ${className}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '20px',
        padding: '10px 20px',
        backgroundColor: '#2c2c2c',
        color: '#fff',
        borderBottom: '2px solid #444',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      {/* Resource Display */}
      {resourceItems.map((resource) => (
        <div
          key={resource.name}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 15px',
            backgroundColor: '#1a1a1a',
            borderRadius: '5px',
            border: `2px solid ${resource.color}`,
          }}
        >
          <span style={{ fontSize: '20px' }}>{resource.icon}</span>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '10px', color: '#888' }}>{resource.name}</span>
            <span style={{ fontSize: '16px', fontWeight: 'bold', color: resource.color }}>
              {Math.floor(resource.value)}
            </span>
          </div>
        </div>
      ))}

      {/* Divider */}
      <div style={{ width: '2px', height: '40px', backgroundColor: '#444' }} />

      {/* Game Controls */}
      <div style={{ display: 'flex', gap: '10px', marginLeft: 'auto' }}>
        {/* Pause/Play Button */}
        <button
          onClick={togglePause}
          style={{
            padding: '8px 16px',
            backgroundColor: isPaused ? '#4CAF50' : '#f44336',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
          }}
        >
          {isPaused ? '▶ Play' : '⏸ Pause'}
        </button>

        {/* Game Speed Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ fontSize: '12px', color: '#888' }}>Speed:</span>
          {[1, 2, 3].map((speed) => (
            <button
              key={speed}
              onClick={() => setGameSpeed(speed)}
              style={{
                padding: '6px 12px',
                backgroundColor: gameSpeed === speed ? '#2196F3' : '#333',
                color: '#fff',
                border: gameSpeed === speed ? '2px solid #64B5F6' : '1px solid #555',
                borderRadius: '3px',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResourceBar;
