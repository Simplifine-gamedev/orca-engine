import React, { useEffect } from 'react';
import { Building } from './buildings/Building';
import { SelectionPanel } from './ui/SelectionPanel';
import { useGameStore, Building as BuildingType } from './store/gameStore';

export const App: React.FC = () => {
  const { buildings } = useGameStore();

  // Initialize with a sample building
  useEffect(() => {
    const sampleBuilding: BuildingType = {
      id: 'barracks-1',
      type: 'Barracks',
      position: { x: 100, y: 100 },
      unitQueue: [],
    };

    const sampleBuilding2: BuildingType = {
      id: 'factory-1',
      type: 'Factory',
      position: { x: 250, y: 100 },
      unitQueue: [],
    };

    const updatedBuildings = new Map<string, BuildingType>();
    updatedBuildings.set(sampleBuilding.id, sampleBuilding);
    updatedBuildings.set(sampleBuilding2.id, sampleBuilding2);
    
    useGameStore.setState({ buildings: updatedBuildings });
  }, []);

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#1a1a1a',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          color: 'white',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '15px',
          borderRadius: '8px',
          fontSize: '14px',
          maxWidth: '300px',
        }}
      >
        <h2 style={{ margin: '0 0 10px 0', fontSize: '18px' }}>Orca RTS - Unit Training</h2>
        <p style={{ margin: '0 0 8px 0', fontSize: '12px', lineHeight: '1.5' }}>
          Click on a building to select it, then use the train buttons.
        </p>
        <p style={{ margin: '0', fontSize: '12px', lineHeight: '1.5', color: '#4CAF50' }}>
          <strong>Hold SHIFT</strong> while clicking to queue 5 units at once!
        </p>
      </div>

      {/* Render all buildings */}
      {Array.from(buildings.values()).map((building) => (
        <Building
          key={building.id}
          buildingId={building.id}
          unitTypes={['Soldier', 'Archer', 'Knight']}
        />
      ))}

      {/* Selection panel at bottom */}
      <SelectionPanel />
    </div>
  );
};

export default App;
