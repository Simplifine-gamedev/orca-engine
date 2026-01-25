import React from 'react';
import { BottomHUD } from './ui/BottomHUD';
import { useGameStore } from './store/gameStore';
import './styles.css';

export const App: React.FC = () => {
  const units = useGameStore((state) => state.units);
  const buildings = useGameStore((state) => state.buildings);
  const selectUnit = useGameStore((state) => state.selectUnit);
  const deselectAllUnits = useGameStore((state) => state.deselectAllUnits);

  const handleCanvasClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Check if clicked on a unit
    const clickedUnit = units.find((unit) => {
      const dx = unit.position.x - x;
      const dy = unit.position.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 20; // 20px click radius
    });

    if (clickedUnit) {
      selectUnit(clickedUnit.id);
    } else {
      deselectAllUnits();
    }
  };

  return (
    <div className="game-container">
      <div className="game-canvas" onClick={handleCanvasClick}>
        {/* Render buildings */}
        {buildings.map((building) => (
          <div
            key={building.id}
            className={`building building-${building.type}`}
            style={{
              left: building.position.x,
              top: building.position.y,
            }}
          >
            🏰
          </div>
        ))}

        {/* Render units */}
        {units.map((unit) => (
          <div
            key={unit.id}
            className={`unit unit-${unit.type} ${unit.isSelected ? 'selected' : ''} ${unit.isIdle ? 'idle' : ''}`}
            style={{
              left: unit.position.x,
              top: unit.position.y,
            }}
          >
            {unit.type === 'worker' && '👷'}
            {unit.type === 'soldier' && '⚔️'}
            {unit.type === 'builder' && '🔨'}
            {unit.isIdle && <div className="idle-badge">💤</div>}
          </div>
        ))}
      </div>

      <BottomHUD />
    </div>
  );
};

export default App;
