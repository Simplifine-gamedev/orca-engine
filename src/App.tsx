import React, { useState, useCallback } from 'react';
import { GameCanvas } from './components/GameCanvas';
import { UnitInfo } from './components/UnitInfo';
import { Unit, UnitTeam, SelectionBox, Vector2 } from './types';
import { getUnitsInBox } from './utils/selection';
import './App.css';

// Generate initial units
const generateUnits = (): Unit[] => {
  const units: Unit[] = [];
  
  // Friendly units (left side)
  for (let i = 0; i < 5; i++) {
    units.push({
      id: `friendly-${i}`,
      name: `Fighter ${i + 1}`,
      team: UnitTeam.FRIENDLY,
      position: { x: 100 + i * 60, y: 200 + (i % 2) * 80 },
      health: 100,
      maxHealth: 100,
      size: 20,
      color: '#4169E1', // Royal Blue
    });
  }
  
  // Enemy units (right side)
  for (let i = 0; i < 5; i++) {
    units.push({
      id: `enemy-${i}`,
      name: `Raider ${i + 1}`,
      team: UnitTeam.ENEMY,
      position: { x: 600 + i * 60, y: 250 + (i % 2) * 80 },
      health: 80,
      maxHealth: 100,
      size: 20,
      color: '#DC143C', // Crimson
    });
  }
  
  // Some neutral units in the middle
  for (let i = 0; i < 3; i++) {
    units.push({
      id: `neutral-${i}`,
      name: `Creature ${i + 1}`,
      team: UnitTeam.NEUTRAL,
      position: { x: 400, y: 150 + i * 100 },
      health: 50,
      maxHealth: 50,
      size: 15,
      color: '#FFD700', // Gold
    });
  }
  
  return units;
};

function App() {
  const [units] = useState<Unit[]>(generateUnits());
  const [selectedUnitIds, setSelectedUnitIds] = useState<string[]>([]);
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>): Vector2 => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);
    
    // Check if clicking on a unit
    const clickedUnit = units.find(unit => {
      const dx = unit.position.x - pos.x;
      const dy = unit.position.y - pos.y;
      return Math.sqrt(dx * dx + dy * dy) <= unit.size;
    });

    if (clickedUnit) {
      // Single unit selection
      if (e.shiftKey) {
        // Add to selection
        setSelectedUnitIds(prev => 
          prev.includes(clickedUnit.id) 
            ? prev.filter(id => id !== clickedUnit.id)
            : [...prev, clickedUnit.id]
        );
      } else {
        // Replace selection
        setSelectedUnitIds([clickedUnit.id]);
      }
    } else {
      // Start marquee selection
      setIsDragging(true);
      setSelectionBox({ start: pos, end: pos });
      
      if (!e.shiftKey) {
        setSelectedUnitIds([]);
      }
    }
  }, [units]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !selectionBox) return;

    const pos = getMousePos(e);
    setSelectionBox(prev => prev ? { ...prev, end: pos } : null);
  }, [isDragging, selectionBox]);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !selectionBox) return;

    // Get all units in the selection box
    const unitsInBox = getUnitsInBox(units, selectionBox);
    const newSelectedIds = unitsInBox.map(u => u.id);

    if (e.shiftKey) {
      // Add to existing selection
      setSelectedUnitIds(prev => {
        const combined = [...prev, ...newSelectedIds];
        return Array.from(new Set(combined));
      });
    } else {
      // Replace selection
      setSelectedUnitIds(newSelectedIds);
    }

    setIsDragging(false);
    setSelectionBox(null);
  }, [isDragging, selectionBox, units]);

  return (
    <div className="App">
      <header>
        <h1>Orca RTS - Unit Selection Demo</h1>
        <div className="instructions">
          <p><strong>Controls:</strong></p>
          <ul>
            <li>Click to select a single unit</li>
            <li>Click and drag to create a marquee selection box</li>
            <li>Hold Shift to add to selection</li>
            <li>Green = Friendly units (can command)</li>
            <li>Red = Enemy units (info only, cannot command)</li>
            <li>Yellow = Neutral units</li>
          </ul>
        </div>
      </header>
      
      <div className="game-container">
        <div className="canvas-container">
          <GameCanvas
            units={units}
            selectedUnitIds={selectedUnitIds}
            selectionBox={selectionBox}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
          />
        </div>
        
        <div className="sidebar">
          <UnitInfo units={units} selectedUnitIds={selectedUnitIds} />
        </div>
      </div>
    </div>
  );
}

export default App;
