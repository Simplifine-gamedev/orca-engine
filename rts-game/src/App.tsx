import React, { useEffect } from 'react';
import { GameScene } from './components/GameScene';
import { useGameStore } from './store/gameStore';

function App() {
  const addUnit = useGameStore(state => state.addUnit);
  const selectUnits = useGameStore(state => state.selectUnits);
  const setUnitDestination = useGameStore(state => state.setUnitDestination);
  
  // Initialize demo units on mount
  useEffect(() => {
    // Create a formation of units
    const unitIds: string[] = [];
    
    for (let i = 0; i < 8; i++) {
      const unitId = `unit-${i}`;
      unitIds.push(unitId);
      
      const row = Math.floor(i / 4);
      const col = i % 4;
      
      addUnit({
        id: unitId,
        position: { x: col * 2, y: 0, z: row * 2 },
        isSelected: false
      });
    }
    
    // Select the first 4 units after a brief delay
    setTimeout(() => {
      const selectedIds = unitIds.slice(0, 4);
      selectUnits(selectedIds);
      
      // Set destinations for selected units after another delay
      setTimeout(() => {
        selectedIds.forEach((id, index) => {
          const destRow = Math.floor(index / 2);
          const destCol = index % 2;
          setUnitDestination(id, { 
            x: 10 + destCol * 2, 
            y: 0, 
            z: 10 + destRow * 2 
          });
        });
      }, 500);
    }, 500);
  }, [addUnit, selectUnits, setUnitDestination]);
  
  return (
    <div className="App">
      <GameScene />
    </div>
  );
}

export default App;
