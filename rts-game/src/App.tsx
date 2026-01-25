import React from 'react';
import { useGameStore } from './store/gameStore';
import { Building } from './buildings/Building';

function App() {
  const { buildings, units } = useGameStore();

  const freeUnits = Object.values(units).filter((unit) => !unit.garrisonedIn);

  return (
    <div style={{ width: '100vw', height: '100vh', backgroundColor: '#2F4F2F', position: 'relative' }}>
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: 15,
          borderRadius: 5,
          maxWidth: 300,
          zIndex: 2000,
        }}
      >
        <h2 style={{ margin: '0 0 10px 0', fontSize: 16 }}>Orca RTS Demo</h2>
        <p style={{ margin: '0 0 5px 0', fontSize: 12 }}>
          Click on a building to see garrisoned units
        </p>
        <p style={{ margin: 0, fontSize: 12 }}>
          Free units on map: {freeUnits.length}
        </p>
      </div>

      {Object.values(buildings).map((building) => (
        <Building key={building.id} building={building} />
      ))}

      {freeUnits.map((unit) => (
        <div
          key={unit.id}
          style={{
            position: 'absolute',
            left: unit.position.x,
            top: unit.position.y,
            width: 30,
            height: 30,
            backgroundColor: '#4169E1',
            border: '2px solid #1E3A8A',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontSize: 10,
            fontWeight: 'bold',
          }}
          title={`${unit.name} (${unit.health}/${unit.maxHealth})`}
        >
          {unit.name[0]}
        </div>
      ))}
    </div>
  );
}

export default App;
