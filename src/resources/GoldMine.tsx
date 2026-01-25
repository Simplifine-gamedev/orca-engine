import React from 'react';
import { Resource } from '../types';
import { useGameStore } from '../store/gameStore';

interface ResourceComponentProps {
  resource: Resource;
}

export const GoldMine: React.FC<ResourceComponentProps> = ({ resource }) => {
  const { selectedResourceId, selectResource } = useGameStore();
  const isSelected = selectedResourceId === resource.id;

  const handleClick = () => {
    selectResource(resource.id);
  };

  const getResourceColor = () => {
    switch (resource.type) {
      case 'goldmine':
        return '#FFD700';
      case 'tree':
        return '#228B22';
      default:
        return '#808080';
    }
  };

  const getResourceIcon = () => {
    switch (resource.type) {
      case 'goldmine':
        return '⛏️';
      case 'tree':
        return '🌲';
      default:
        return '❓';
    }
  };

  return (
    <div
      onClick={handleClick}
      style={{
        position: 'absolute',
        left: `${resource.position.x}px`,
        top: `${resource.position.y}px`,
        width: '60px',
        height: '60px',
        backgroundColor: getResourceColor(),
        border: isSelected ? '3px solid white' : '2px solid #333',
        borderRadius: '8px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: isSelected ? '0 0 15px rgba(255,255,255,0.8)' : '0 2px 4px rgba(0,0,0,0.3)',
        transition: 'all 0.2s ease',
        userSelect: 'none'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'scale(1.1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
      }}
    >
      <div style={{ fontSize: '24px' }}>{getResourceIcon()}</div>
      {resource.workersAssigned > 0 && (
        <div style={{
          position: 'absolute',
          top: '-8px',
          right: '-8px',
          backgroundColor: '#FF4444',
          color: 'white',
          borderRadius: '50%',
          width: '20px',
          height: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '12px',
          fontWeight: 'bold',
          border: '2px solid white'
        }}>
          {resource.workersAssigned}
        </div>
      )}
    </div>
  );
};

export const TreeResource: React.FC<ResourceComponentProps> = ({ resource }) => {
  return <GoldMine resource={resource} />;
};
