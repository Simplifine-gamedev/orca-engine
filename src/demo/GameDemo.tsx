// Demo application showing auto-opening gates
import React, { useEffect, useState } from 'react';
import { WallSystem } from '../buildings/WallSystem';
import { wallStore } from '../store/wallStore';
import { unitManager } from '../utils/unitManager';
import { Position, Unit } from '../types';

const TILE_SIZE = 32;
const GRID_WIDTH = 30;
const GRID_HEIGHT = 20;

export const GameDemo: React.FC = () => {
  const [units, setUnits] = useState<Unit[]>([]);

  useEffect(() => {
    // Initialize the demo
    initializeDemo();

    // Subscribe to unit updates
    const unsubscribe = wallStore.subscribe(() => {
      setUnits(unitManager.getUnits());
    });

    // Start systems
    unitManager.startMovementUpdates(300);

    return () => {
      unsubscribe();
      unitManager.stopMovementUpdates();
    };
  }, []);

  const initializeDemo = () => {
    // Clear existing data
    wallStore.reset();
    unitManager.reset();

    // Create a wall with a gate in the middle
    // Horizontal wall
    for (let x = 5; x < 25; x++) {
      if (x === 15) {
        // Gate in the middle
        wallStore.addWall({
          id: `gate-${x}-10`,
          position: { x, y: 10 },
          type: 'gate',
          team: 'friendly',
        });
      } else {
        wallStore.addWall({
          id: `wall-${x}-10`,
          position: { x, y: 10 },
          type: 'wall',
          team: 'friendly',
        });
      }
    }

    // Create some friendly units
    const friendlyUnits = [
      { id: 'unit-1', position: { x: 10, y: 5 } },
      { id: 'unit-2', position: { x: 12, y: 6 } },
      { id: 'unit-3', position: { x: 8, y: 7 } },
    ];

    friendlyUnits.forEach((data) => {
      unitManager.createUnit(data.id, data.position, 'friendly');
    });

    // Create an enemy unit
    unitManager.createUnit('enemy-1', { x: 20, y: 5 }, 'enemy');
  };

  const handleMoveUnits = () => {
    // Move friendly units through the gate
    const target: Position = { x: 15, y: 15 };
    unitManager.getUnits().forEach((unit) => {
      if (unit.team === 'friendly') {
        unitManager.moveUnit(unit.id, target);
      }
    });
  };

  const handleMoveBack = () => {
    // Move friendly units back
    const target: Position = { x: 10, y: 5 };
    unitManager.getUnits().forEach((unit) => {
      if (unit.team === 'friendly') {
        unitManager.moveUnit(unit.id, target);
      }
    });
  };

  const handleReset = () => {
    initializeDemo();
  };

  const renderUnits = () => {
    return units.map((unit) => {
      const style: React.CSSProperties = {
        position: 'absolute',
        left: unit.position.x * TILE_SIZE,
        top: unit.position.y * TILE_SIZE,
        width: TILE_SIZE - 4,
        height: TILE_SIZE - 4,
        borderRadius: '50%',
        backgroundColor: unit.team === 'friendly' ? '#3182ce' : '#e53e3e',
        border: '2px solid #fff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '16px',
        transition: 'all 0.3s ease-in-out',
        zIndex: 10,
      };

      return (
        <div key={unit.id} style={style}>
          {unit.team === 'friendly' ? '🛡' : '⚔'}
        </div>
      );
    });
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Auto-Opening Gates Demo</h1>
      <p>
        Friendly units (blue with shields) will automatically open gates when they approach.
        Enemy units (red with swords) cannot pass through gates.
      </p>

      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={handleMoveUnits}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: '#3182ce',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Move Units Through Gate
        </button>
        <button
          onClick={handleMoveBack}
          style={{
            padding: '10px 20px',
            marginRight: '10px',
            backgroundColor: '#38a169',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Move Units Back
        </button>
        <button
          onClick={handleReset}
          style={{
            padding: '10px 20px',
            backgroundColor: '#718096',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Reset
        </button>
      </div>

      <div style={{ position: 'relative', display: 'inline-block' }}>
        <WallSystem
          tileSize={TILE_SIZE}
          gridWidth={GRID_WIDTH}
          gridHeight={GRID_HEIGHT}
        />
        {renderUnits()}
      </div>

      <div style={{ marginTop: '20px', maxWidth: '600px' }}>
        <h2>Features Implemented:</h2>
        <ul>
          <li>✅ Auto-detect friendly units near gates</li>
          <li>✅ Automatic gate opening animation</li>
          <li>✅ Pathfinding through open gates</li>
          <li>✅ Auto-close with delay after units pass</li>
          <li>✅ Gates stay closed for enemy units</li>
        </ul>
      </div>
    </div>
  );
};

export default GameDemo;
