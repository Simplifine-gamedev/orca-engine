import React, { useEffect } from 'react';
import { useGameStore } from '../store/gameStore';
import { GoldMine } from '../resources/GoldMine';
import { Tree } from '../resources/Tree';
import { SelectionPanel } from '../ui/SelectionPanel';
import { ResourceData } from '../types/resource';

export const Game: React.FC = () => {
  const { resources, addResource, deselectEntity } = useGameStore();
  
  // Initialize demo resources
  useEffect(() => {
    const demoResources: ResourceData[] = [
      {
        id: 'gold-1',
        type: 'gold_mine',
        name: 'Gold Mine Alpha',
        amountRemaining: 5000,
        maxAmount: 5000,
        workersAssigned: 0,
        maxWorkers: 5,
        gatherRate: 10,
        position: { x: 100, y: 150 }
      },
      {
        id: 'gold-2',
        type: 'gold_mine',
        name: 'Gold Mine Beta',
        amountRemaining: 3200,
        maxAmount: 5000,
        workersAssigned: 2,
        maxWorkers: 5,
        gatherRate: 10,
        position: { x: 300, y: 200 }
      },
      {
        id: 'tree-1',
        type: 'tree',
        name: 'Forest Grove',
        amountRemaining: 800,
        maxAmount: 1000,
        workersAssigned: 1,
        maxWorkers: 3,
        gatherRate: 5,
        position: { x: 500, y: 180 }
      },
      {
        id: 'tree-2',
        type: 'tree',
        name: 'Ancient Oak',
        amountRemaining: 1000,
        maxAmount: 1000,
        workersAssigned: 0,
        maxWorkers: 3,
        gatherRate: 5,
        position: { x: 200, y: 350 }
      },
      {
        id: 'gold-3',
        type: 'gold_mine',
        name: 'Gold Mine Gamma',
        amountRemaining: 4500,
        maxAmount: 5000,
        workersAssigned: 3,
        maxWorkers: 5,
        gatherRate: 10,
        position: { x: 450, y: 380 }
      }
    ];
    
    demoResources.forEach(resource => addResource(resource));
  }, [addResource]);
  
  const handleBackgroundClick = () => {
    deselectEntity();
  };
  
  return (
    <div
      onClick={handleBackgroundClick}
      style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#2a5a3a',
        backgroundImage: `
          repeating-linear-gradient(
            0deg,
            transparent,
            transparent 50px,
            rgba(0, 0, 0, 0.05) 50px,
            rgba(0, 0, 0, 0.05) 100px
          ),
          repeating-linear-gradient(
            90deg,
            transparent,
            transparent 50px,
            rgba(0, 0, 0, 0.05) 50px,
            rgba(0, 0, 0, 0.05) 100px
          )
        `,
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      {/* Title/Instructions */}
      <div
        style={{
          position: 'absolute',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '16px 24px',
          borderRadius: '8px',
          textAlign: 'center',
          zIndex: 100
        }}
      >
        <h1 style={{ margin: '0 0 8px 0', fontSize: '24px' }}>Orca RTS - Resource Selection Demo</h1>
        <p style={{ margin: 0, fontSize: '14px', color: '#ccc' }}>
          Click on resources to view details and manage workers
        </p>
      </div>
      
      {/* Resources */}
      {resources.map((resource) => {
        if (resource.type === 'gold_mine') {
          return <GoldMine key={resource.id} resource={resource} />;
        } else if (resource.type === 'tree') {
          return <Tree key={resource.id} resource={resource} />;
        }
        return null;
      })}
      
      {/* Selection Panel */}
      <SelectionPanel />
    </div>
  );
};
