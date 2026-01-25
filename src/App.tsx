import React, { useState, useCallback } from 'react';
import RTSUnit from './units/RTSUnit';
import NeutralMob from './units/NeutralMob';
import './styles.css';

export type UnitType = 'player' | 'enemy' | 'neutral';
export type CursorType = 'default' | 'attack' | 'friendly' | 'move';

interface Unit {
  id: string;
  x: number;
  y: number;
  type: UnitType;
  health: number;
  maxHealth: number;
}

function App() {
  const [cursorType, setCursorType] = useState<CursorType>('default');
  const [hoveredUnit, setHoveredUnit] = useState<string | null>(null);
  const [selectedUnit, setSelectedUnit] = useState<string | null>(null);

  // Initialize some units for demonstration
  const [units] = useState<Unit[]>([
    { id: 'player1', x: 100, y: 150, type: 'player', health: 100, maxHealth: 100 },
    { id: 'player2', x: 150, y: 200, type: 'player', health: 80, maxHealth: 100 },
    { id: 'enemy1', x: 400, y: 150, type: 'enemy', health: 90, maxHealth: 100 },
    { id: 'enemy2', x: 450, y: 200, type: 'enemy', health: 100, maxHealth: 100 },
    { id: 'neutral1', x: 250, y: 300, type: 'neutral', health: 60, maxHealth: 100 },
    { id: 'neutral2', x: 350, y: 350, type: 'neutral', health: 70, maxHealth: 100 },
  ]);

  const handleUnitHover = useCallback((unitId: string | null, unitType: UnitType | null) => {
    setHoveredUnit(unitId);
    
    if (!unitId || !unitType) {
      setCursorType('default');
      return;
    }

    // Determine cursor type based on unit type
    if (selectedUnit) {
      // If a unit is selected, show attack cursor for enemies
      if (unitType === 'enemy' || unitType === 'neutral') {
        setCursorType('attack');
      } else if (unitType === 'player') {
        setCursorType('friendly');
      }
    } else {
      // If no unit is selected, just hovering
      if (unitType === 'enemy' || unitType === 'neutral') {
        setCursorType('attack');
      } else if (unitType === 'player') {
        setCursorType('friendly');
      }
    }
  }, [selectedUnit]);

  const handleUnitClick = useCallback((unitId: string, unitType: UnitType) => {
    if (unitType === 'player') {
      // Select friendly units
      setSelectedUnit(unitId);
      console.log(`Selected unit: ${unitId}`);
    } else {
      // Attack enemy or neutral units
      if (selectedUnit) {
        console.log(`Unit ${selectedUnit} attacking ${unitId}`);
      } else {
        console.log(`Cannot attack ${unitId} without selecting a unit first`);
      }
    }
  }, [selectedUnit]);

  const handleMapClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    // Only handle clicks on the map background, not on units
    if (e.target === e.currentTarget) {
      if (selectedUnit) {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        console.log(`Unit ${selectedUnit} moving to (${x}, ${y})`);
      }
      setSelectedUnit(null);
      setCursorType('default');
    }
  }, [selectedUnit]);

  const handleMapHover = useCallback(() => {
    if (!hoveredUnit) {
      setCursorType(selectedUnit ? 'move' : 'default');
    }
  }, [hoveredUnit, selectedUnit]);

  return (
    <div className="app">
      <div className="header">
        <h1>RTS Combat Demo</h1>
        <div className="controls">
          <p>
            <strong>Instructions:</strong> Click on friendly units (blue) to select them. 
            Hover over enemy units (red) or neutral mobs (yellow) to see attack cursor.
            Click on map to move selected unit.
          </p>
          <div className="status">
            <span>Cursor: <strong>{cursorType}</strong></span>
            {selectedUnit && <span> | Selected: <strong>{selectedUnit}</strong></span>}
            {hoveredUnit && <span> | Hovering: <strong>{hoveredUnit}</strong></span>}
          </div>
        </div>
      </div>
      
      <div 
        className={`game-map cursor-${cursorType}`}
        onClick={handleMapClick}
        onMouseMove={handleMapHover}
      >
        {units.map(unit => {
          const isHovered = hoveredUnit === unit.id;
          const isSelected = selectedUnit === unit.id;

          if (unit.type === 'neutral') {
            return (
              <NeutralMob
                key={unit.id}
                id={unit.id}
                x={unit.x}
                y={unit.y}
                health={unit.health}
                maxHealth={unit.maxHealth}
                isHovered={isHovered}
                onHover={handleUnitHover}
                onClick={handleUnitClick}
              />
            );
          }

          return (
            <RTSUnit
              key={unit.id}
              id={unit.id}
              x={unit.x}
              y={unit.y}
              type={unit.type}
              health={unit.health}
              maxHealth={unit.maxHealth}
              isHovered={isHovered}
              isSelected={isSelected}
              onHover={handleUnitHover}
              onClick={handleUnitClick}
            />
          );
        })}
      </div>
    </div>
  );
}

export default App;
