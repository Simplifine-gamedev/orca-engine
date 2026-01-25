import React from 'react';
import { useGameStore } from '../store/gameStore';
import { ResourceData } from '../types/resource';

interface TreeProps {
  resource: ResourceData;
  style?: React.CSSProperties;
}

export const Tree: React.FC<TreeProps> = ({ resource, style }) => {
  const { selectEntity, selectedEntity } = useGameStore();
  
  const isSelected = selectedEntity?.id === resource.id;
  
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    selectEntity({
      id: resource.id,
      type: 'resource',
      data: resource
    });
  };
  
  const fillPercentage = (resource.amountRemaining / resource.maxAmount) * 100;
  
  return (
    <div
      onClick={handleClick}
      style={{
        position: 'absolute',
        left: resource.position.x,
        top: resource.position.y,
        width: '80px',
        height: '80px',
        cursor: 'pointer',
        userSelect: 'none',
        ...style
      }}
      className={`tree ${isSelected ? 'selected' : ''}`}
    >
      <div
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#228B22',
          borderRadius: '8px',
          border: isSelected ? '3px solid #00FF00' : '2px solid #006400',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: isSelected 
            ? '0 0 15px rgba(0, 255, 0, 0.5)'
            : '0 4px 8px rgba(0, 0, 0, 0.3)',
          transition: 'all 0.2s ease',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {/* Resource fill indicator */}
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: `${fillPercentage}%`,
            backgroundColor: 'rgba(34, 139, 34, 0.4)',
            transition: 'height 0.3s ease',
            zIndex: 0
          }}
        />
        
        {/* Icon/Content */}
        <div style={{ position: 'relative', zIndex: 1, textAlign: 'center' }}>
          <div style={{ fontSize: '32px', marginBottom: '4px' }}>🌲</div>
          <div 
            style={{ 
              fontSize: '10px', 
              fontWeight: 'bold',
              color: 'white',
              textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)'
            }}
          >
            {resource.amountRemaining}
          </div>
        </div>
        
        {/* Workers indicator */}
        {resource.workersAssigned > 0 && (
          <div
            style={{
              position: 'absolute',
              top: '4px',
              right: '4px',
              backgroundColor: 'rgba(0, 0, 0, 0.6)',
              borderRadius: '50%',
              width: '20px',
              height: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '10px',
              fontWeight: 'bold',
              color: 'white',
              zIndex: 2
            }}
          >
            {resource.workersAssigned}
          </div>
        )}
      </div>
    </div>
  );
};
