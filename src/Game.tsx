import React from 'react';
import { useGameStore } from './store/gameStore';
import { GoldMine, TreeResource } from './resources/GoldMine';
import { SelectionPanel } from './ui/SelectionPanel';

export const Game: React.FC = () => {
  const { resources, selectResource } = useGameStore();

  const handleBackgroundClick = () => {
    selectResource(null);
  };

  return (
    <div 
      style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#2C5F2D',
        backgroundImage: 'linear-gradient(45deg, #2C5F2D 25%, #3A7A3D 25%, #3A7A3D 50%, #2C5F2D 50%, #2C5F2D 75%, #3A7A3D 75%, #3A7A3D)',
        backgroundSize: '40px 40px',
        position: 'relative',
        overflow: 'hidden',
        fontFamily: 'Arial, sans-serif'
      }}
      onClick={handleBackgroundClick}
    >
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '15px 20px',
        borderRadius: '8px',
        fontSize: '16px',
        fontWeight: 'bold',
        border: '2px solid #FFD700'
      }}>
        Orca RTS - Resource Management
      </div>

      <div style={{
        position: 'absolute',
        top: '70px',
        left: '20px',
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        color: 'white',
        padding: '10px 15px',
        borderRadius: '6px',
        fontSize: '12px',
        maxWidth: '300px',
        lineHeight: '1.4'
      }}>
        Click on resources (gold mines or trees) to view their information panel
      </div>

      {resources.map((resource) => {
        if (resource.type === 'goldmine') {
          return <GoldMine key={resource.id} resource={resource} />;
        } else if (resource.type === 'tree') {
          return <TreeResource key={resource.id} resource={resource} />;
        }
        return null;
      })}

      <SelectionPanel />
    </div>
  );
};
