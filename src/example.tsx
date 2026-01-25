/**
 * Example usage of the auto-opening gates system
 */

import React, { useEffect, useState } from 'react';
import { WallSystem, WallSystemUtils } from './buildings/WallSystem';
import { findPath, smoothPath } from './pathfinding/pathfinding';
import { wallStore, Position } from './store/wallStore';

const GRID_SIZE = 30;
const CELL_SIZE = 32;

export function RTSGameExample() {
  const [selectedUnitId, setSelectedUnitId] = useState<string | null>(null);
  const [unitPaths, setUnitPaths] = useState<Map<string, Position[]>>(new Map());

  useEffect(() => {
    // Initialize game world
    initializeWorld();

    // Cleanup on unmount
    return () => {
      wallStore.reset();
    };
  }, []);

  const initializeWorld = () => {
    // Create walls to form a compound
    createWallPerimeter();

    // Create gates in the walls
    WallSystemUtils.createGate({ x: 15, y: 10 }, 'player1'); // North gate
    WallSystemUtils.createGate({ x: 15, y: 20 }, 'player1'); // South gate

    // Create some friendly units
    WallSystemUtils.createUnit({ x: 15, y: 5 }, 'player1');
    WallSystemUtils.createUnit({ x: 12, y: 15 }, 'player1');
    WallSystemUtils.createUnit({ x: 18, y: 15 }, 'player1');

    // Create enemy units (different owner)
    WallSystemUtils.createUnit({ x: 5, y: 5 }, 'player2');
    WallSystemUtils.createUnit({ x: 25, y: 25 }, 'player2');
  };

  const createWallPerimeter = () => {
    // Create horizontal walls
    for (let x = 10; x <= 20; x++) {
      if (x !== 15) { // Leave space for gates
        wallStore.addWall({ id: `wall_top_${x}`, position: { x, y: 10 }, ownerId: 'player1' });
        wallStore.addWall({ id: `wall_bottom_${x}`, position: { x, y: 20 }, ownerId: 'player1' });
      }
    }

    // Create vertical walls
    for (let y = 11; y < 20; y++) {
      wallStore.addWall({ id: `wall_left_${y}`, position: { x: 10, y }, ownerId: 'player1' });
      wallStore.addWall({ id: `wall_right_${y}`, position: { x: 20, y }, ownerId: 'player1' });
    }
  };

  const handleMapClick = (x: number, y: number) => {
    if (!selectedUnitId) return;

    const unit = wallStore.getUnit(selectedUnitId);
    if (!unit) return;

    // Calculate path
    const path = findPath(
      unit.position,
      { x, y },
      {
        unitOwnerId: unit.ownerId,
        gridWidth: GRID_SIZE,
        gridHeight: GRID_SIZE,
        allowDiagonal: true,
      }
    );

    if (path) {
      const smoothedPath = smoothPath(path);
      setUnitPaths(new Map(unitPaths.set(selectedUnitId, smoothedPath)));
      
      // Start moving unit along path
      moveUnitAlongPath(selectedUnitId, smoothedPath);
    }
  };

  const moveUnitAlongPath = async (unitId: string, path: Position[]) => {
    for (let i = 1; i < path.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 200));
      WallSystemUtils.moveUnit(unitId, path[i]);

      // Update path display
      const newPaths = new Map(unitPaths);
      newPaths.set(unitId, path.slice(i));
      setUnitPaths(newPaths);
    }

    // Clear path when done
    const newPaths = new Map(unitPaths);
    newPaths.delete(unitId);
    setUnitPaths(newPaths);
  };

  const renderGrid = () => {
    const cells = [];
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        cells.push(
          <div
            key={`${x}-${y}`}
            onClick={() => handleMapClick(x, y)}
            style={{
              position: 'absolute',
              left: x * CELL_SIZE,
              top: y * CELL_SIZE,
              width: CELL_SIZE,
              height: CELL_SIZE,
              border: '1px solid #333',
              backgroundColor: '#1a1a1a',
              cursor: 'pointer',
            }}
          />
        );
      }
    }
    return cells;
  };

  const renderUnits = () => {
    const units = wallStore.getAllUnits();
    return units.map((unit) => (
      <div
        key={unit.id}
        onClick={(e) => {
          e.stopPropagation();
          setSelectedUnitId(unit.id);
        }}
        style={{
          position: 'absolute',
          left: unit.position.x * CELL_SIZE + 4,
          top: unit.position.y * CELL_SIZE + 4,
          width: CELL_SIZE - 8,
          height: CELL_SIZE - 8,
          borderRadius: '50%',
          backgroundColor: unit.ownerId === 'player1' ? '#4CAF50' : '#F44336',
          border: selectedUnitId === unit.id ? '3px solid yellow' : '2px solid #333',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '16px',
        }}
        title={`Unit ${unit.id} (${unit.ownerId})`}
      >
        {unit.ownerId === 'player1' ? '👤' : '👾'}
      </div>
    ));
  };

  const renderWalls = () => {
    const walls = wallStore.getAllWalls();
    return walls.map((wall) => (
      <div
        key={wall.id}
        style={{
          position: 'absolute',
          left: wall.position.x * CELL_SIZE,
          top: wall.position.y * CELL_SIZE,
          width: CELL_SIZE,
          height: CELL_SIZE,
          backgroundColor: '#795548',
          border: '2px solid #333',
        }}
      />
    ));
  };

  const renderPaths = () => {
    const pathElements: React.ReactNode[] = [];
    
    unitPaths.forEach((path, unitId) => {
      for (let i = 0; i < path.length - 1; i++) {
        const from = path[i];
        const to = path[i + 1];
        
        pathElements.push(
          <line
            key={`${unitId}-${i}`}
            x1={from.x * CELL_SIZE + CELL_SIZE / 2}
            y1={from.y * CELL_SIZE + CELL_SIZE / 2}
            x2={to.x * CELL_SIZE + CELL_SIZE / 2}
            y2={to.y * CELL_SIZE + CELL_SIZE / 2}
            stroke="#FFD700"
            strokeWidth="2"
            strokeDasharray="4,4"
          />
        );
      }
    });

    return (
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: GRID_SIZE * CELL_SIZE,
          height: GRID_SIZE * CELL_SIZE,
          pointerEvents: 'none',
        }}
      >
        {pathElements}
      </svg>
    );
  };

  return (
    <div style={{ padding: '20px', backgroundColor: '#111', minHeight: '100vh' }}>
      <div style={{ marginBottom: '20px', color: 'white' }}>
        <h1>RTS Auto-Opening Gates Demo</h1>
        <div style={{ marginTop: '10px' }}>
          <h3>Instructions:</h3>
          <ul>
            <li>🟢 Green units are friendly (Player 1)</li>
            <li>🔴 Red units are enemies (Player 2)</li>
            <li>🚪 Gates open automatically for friendly units</li>
            <li>Click a unit to select it</li>
            <li>Click the map to move selected unit</li>
            <li>Watch gates open as friendly units approach!</li>
          </ul>
        </div>
      </div>

      <div
        style={{
          position: 'relative',
          width: GRID_SIZE * CELL_SIZE,
          height: GRID_SIZE * CELL_SIZE,
          border: '2px solid #666',
        }}
      >
        {renderGrid()}
        {renderWalls()}
        {renderPaths()}
        <WallSystem 
          detectionRadius={3.0}
          closeDelay={2000}
          updateInterval={100}
        />
        {renderUnits()}
      </div>
    </div>
  );
}
