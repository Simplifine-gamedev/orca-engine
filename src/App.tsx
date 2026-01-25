import React, { useEffect } from 'react';
import TreeSystem from './resources/TreeSystem';
import ResourceBar from './ui/ResourceBar';
import LumberCamp from './resources/LumberCamp';
import { useGameStore } from './store/gameStore';
import { Worker, WORKER_CARRY_CAPACITY } from './types';

const App: React.FC = () => {
  const addWorker = useGameStore((state) => state.addWorker);
  const workers = useGameStore((state) => state.workers);

  // Initialize some workers on mount
  useEffect(() => {
    if (workers.length === 0) {
      // Add 3 starting workers
      for (let i = 0; i < 3; i++) {
        const worker: Worker = {
          id: `worker_${i}`,
          position: { x: 100 + i * 50, y: 100 },
          isGathering: false,
          targetTreeId: null,
          carryingWood: 0,
          maxCarryCapacity: WORKER_CARRY_CAPACITY,
        };
        addWorker(worker);
      }
    }
  }, []);

  const handleAddWorker = () => {
    const newWorker: Worker = {
      id: `worker_${Date.now()}`,
      position: { x: Math.random() * 600 + 100, y: Math.random() * 400 + 100 },
      isGathering: false,
      targetTreeId: null,
      carryingWood: 0,
      maxCarryCapacity: WORKER_CARRY_CAPACITY,
    };
    addWorker(newWorker);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Resource Bar at the top */}
      <ResourceBar />

      {/* Main Game Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Sidebar - Building Menu */}
        <div 
          style={{ 
            width: '300px', 
            backgroundColor: '#1a1a1a', 
            padding: '20px',
            overflowY: 'auto',
            borderRight: '2px solid #333',
          }}
        >
          <h2 style={{ color: '#fff', marginBottom: '20px', fontSize: '20px' }}>
            Buildings & Units
          </h2>
          
          {/* Worker Management */}
          <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#2c2c2c', borderRadius: '8px' }}>
            <h3 style={{ color: '#fff', marginBottom: '10px', fontSize: '16px' }}>
              Workers ({workers.length})
            </h3>
            <button
              onClick={handleAddWorker}
              style={{
                padding: '10px 20px',
                backgroundColor: '#2196F3',
                color: '#fff',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold',
                width: '100%',
              }}
            >
              Train Worker (50 food)
            </button>
            <div style={{ marginTop: '10px', fontSize: '12px', color: '#888' }}>
              Click a worker, then click a tree to gather wood
            </div>
          </div>

          {/* Lumber Camp Manager */}
          <LumberCamp />

          {/* Instructions */}
          <div 
            style={{ 
              marginTop: '20px',
              padding: '15px',
              backgroundColor: '#2c2c2c',
              borderRadius: '8px',
              color: '#ccc',
              fontSize: '13px',
            }}
          >
            <h3 style={{ color: '#fff', marginBottom: '10px', fontSize: '16px' }}>
              How to Play
            </h3>
            <ul style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
              <li>Click a worker to select it</li>
              <li>Click a tree to send the worker to gather wood</li>
              <li>Workers will automatically return when full</li>
              <li>Build lumber camps near forests for bonuses</li>
              <li>Trees will regrow over time</li>
            </ul>
          </div>
        </div>

        {/* Main Game Canvas */}
        <div 
          style={{ 
            flex: 1, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            backgroundColor: '#0a0a0a',
            padding: '20px',
          }}
        >
          <TreeSystem mapWidth={800} mapHeight={600} treeCount={50} />
        </div>
      </div>
    </div>
  );
};

export default App;
