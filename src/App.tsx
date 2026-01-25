import React, { useState, useCallback } from 'react';
import RTSUnit from './units/RTSUnit';
import NeutralMob from './units/NeutralMob';
import { Unit, CursorMode } from './types';
import './styles/cursor.css';

const App: React.FC = () => {
  // Sample units for demonstration
  const [units, setUnits] = useState<Unit[]>([
    {
      id: 'friendly-1',
      type: 'friendly',
      position: { x: 100, y: 150 },
      health: 100,
      maxHealth: 100,
      name: 'Knight',
    },
    {
      id: 'friendly-2',
      type: 'friendly',
      position: { x: 200, y: 150 },
      health: 80,
      maxHealth: 100,
      name: 'Archer',
    },
    {
      id: 'enemy-1',
      type: 'enemy',
      position: { x: 500, y: 200 },
      health: 100,
      maxHealth: 100,
      name: 'Orc Warrior',
    },
    {
      id: 'enemy-2',
      type: 'enemy',
      position: { x: 600, y: 250 },
      health: 75,
      maxHealth: 100,
      name: 'Goblin',
    },
    {
      id: 'neutral-1',
      type: 'neutral',
      position: { x: 350, y: 400 },
      health: 120,
      maxHealth: 120,
      name: 'Forest Wolf',
    },
    {
      id: 'neutral-2',
      type: 'neutral',
      position: { x: 550, y: 450 },
      health: 150,
      maxHealth: 150,
      name: 'Ancient Bear',
    },
  ]);

  const [selectedUnit, setSelectedUnit] = useState<Unit | null>(null);
  const [hoveredUnit, setHoveredUnit] = useState<Unit | null>(null);
  const [cursorMode, setCursorMode] = useState<CursorMode>('default');

  // Handle unit hover to change cursor
  const handleUnitHover = useCallback((unit: Unit | null) => {
    setHoveredUnit(unit);
    
    if (!unit) {
      setCursorMode('move');
      return;
    }

    // Determine cursor mode based on unit type
    switch (unit.type) {
      case 'enemy':
        setCursorMode('attack-enemy');
        break;
      case 'neutral':
        setCursorMode('attack-neutral');
        break;
      case 'friendly':
        setCursorMode('friendly');
        break;
      default:
        setCursorMode('move');
    }
  }, []);

  // Handle unit click
  const handleUnitClick = useCallback((unit: Unit) => {
    if (selectedUnit && unit.type !== 'friendly') {
      // Attack command
      console.log(`${selectedUnit.name} attacking ${unit.name}`);
      
      // Simulate attack - reduce health
      setUnits(prev => prev.map(u => 
        u.id === unit.id ? { ...u, health: Math.max(0, u.health - 25) } : u
      ));
    } else if (unit.type === 'friendly') {
      // Select friendly unit
      setSelectedUnit(unit);
      setUnits(prev => prev.map(u => ({
        ...u,
        isSelected: u.id === unit.id,
      })));
    }
  }, [selectedUnit]);

  // Handle canvas click (move command)
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (selectedUnit) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      console.log(`${selectedUnit.name} moving to (${Math.floor(x)}, ${Math.floor(y)})`);
      
      // Update unit position
      setUnits(prev => prev.map(u => 
        u.id === selectedUnit.id ? { ...u, position: { x, y } } : u
      ));
    }
  }, [selectedUnit]);

  // Get cursor class based on mode
  const getCursorClass = () => {
    switch (cursorMode) {
      case 'attack-enemy':
        return 'cursor-attack-enemy';
      case 'attack-neutral':
        return 'cursor-attack-neutral';
      case 'friendly':
        return 'cursor-friendly';
      case 'move':
        return 'cursor-move';
      default:
        return 'cursor-move';
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      {/* Game UI */}
      <div className="game-ui">
        <h2>Orca RTS - Combat Demo</h2>
        <p><strong>Controls:</strong></p>
        <p>• Click friendly unit to select</p>
        <p>• Click enemy/neutral to attack</p>
        <p>• Click ground to move</p>
        <p style={{ marginTop: '10px' }}>
          <strong>Cursor Feedback:</strong>
        </p>
        <p>🎯 Red crosshair = Enemy</p>
        <p>⭐ Orange star = Neutral mob</p>
        <p>🟢 Green circle = Friendly</p>
        <p>➕ Blue cross = Move</p>
        {selectedUnit && (
          <p style={{ marginTop: '10px', color: '#4CAF50' }}>
            <strong>Selected:</strong> {selectedUnit.name}
          </p>
        )}
        {hoveredUnit && (
          <p style={{ color: '#FFC107' }}>
            <strong>Hover:</strong> {hoveredUnit.name} ({hoveredUnit.type})
          </p>
        )}
      </div>

      {/* Game Canvas */}
      <div
        className={`game-canvas ${getCursorClass()}`}
        onClick={handleCanvasClick}
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#2c3e50',
          backgroundImage: 'linear-gradient(45deg, #34495e 25%, transparent 25%), linear-gradient(-45deg, #34495e 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #34495e 75%), linear-gradient(-45deg, transparent 75%, #34495e 75%)',
          backgroundSize: '40px 40px',
          backgroundPosition: '0 0, 0 20px, 20px -20px, -20px 0px',
          position: 'relative',
        }}
      >
        {/* Render Units */}
        {units.map((unit) => {
          if (unit.type === 'neutral') {
            return (
              <NeutralMob
                key={unit.id}
                unit={unit}
                onHover={handleUnitHover}
                onClick={handleUnitClick}
              />
            );
          } else {
            return (
              <RTSUnit
                key={unit.id}
                unit={unit}
                onHover={handleUnitHover}
                onClick={handleUnitClick}
              />
            );
          }
        })}
      </div>
    </div>
  );
};

export default App;
