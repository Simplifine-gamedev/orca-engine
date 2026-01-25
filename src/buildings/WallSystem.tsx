// Wall and Gate System Component
import React, { useEffect, useState, useCallback } from 'react';
import { wallStore } from '../store/wallStore';
import { Wall, Gate, Unit, Position } from '../types';

interface WallSystemProps {
  tileSize?: number; // Size of each tile in pixels
  gridWidth?: number;
  gridHeight?: number;
}

export const WallSystem: React.FC<WallSystemProps> = ({
  tileSize = 32,
  gridWidth = 50,
  gridHeight = 50,
}) => {
  const [walls, setWalls] = useState<Wall[]>([]);
  const [gates, setGates] = useState<Gate[]>([]);
  const [animatingGates, setAnimatingGates] = useState<Set<string>>(new Set());

  // Subscribe to store updates
  useEffect(() => {
    const unsubscribe = wallStore.subscribe((state) => {
      setWalls(wallStore.getWalls());
      setGates(wallStore.getGates());
    });

    // Initial state
    setWalls(wallStore.getWalls());
    setGates(wallStore.getGates());

    // Start automatic gate checking
    wallStore.startGateChecking(100);

    return () => {
      unsubscribe();
      wallStore.stopGateChecking();
    };
  }, []);

  // Trigger animation when gate state changes
  useEffect(() => {
    gates.forEach((gate) => {
      // Add animation class when gate opens or closes
      setAnimatingGates((prev) => new Set(prev).add(gate.id));
      
      // Remove animation class after animation completes
      setTimeout(() => {
        setAnimatingGates((prev) => {
          const newSet = new Set(prev);
          newSet.delete(gate.id);
          return newSet;
        });
      }, 300); // Animation duration
    });
  }, [gates.map(g => `${g.id}-${g.isOpen}`).join(',')]);

  // Render a wall
  const renderWall = (wall: Wall) => {
    const style: React.CSSProperties = {
      position: 'absolute',
      left: wall.position.x * tileSize,
      top: wall.position.y * tileSize,
      width: tileSize,
      height: tileSize,
      backgroundColor: wall.team === 'friendly' ? '#4a5568' : '#718096',
      border: '1px solid #2d3748',
      boxSizing: 'border-box',
    };

    return <div key={wall.id} style={style} className="wall" />;
  };

  // Render a gate with animation
  const renderGate = (gate: Gate) => {
    const isAnimating = animatingGates.has(gate.id);
    
    const style: React.CSSProperties = {
      position: 'absolute',
      left: gate.position.x * tileSize,
      top: gate.position.y * tileSize,
      width: tileSize,
      height: tileSize,
      backgroundColor: gate.isOpen ? '#48bb78' : '#ed8936',
      border: '2px solid #2d3748',
      boxSizing: 'border-box',
      transition: isAnimating ? 'all 0.3s ease-in-out' : 'none',
      opacity: gate.isOpen ? 0.6 : 1,
      transform: gate.isOpen ? 'scale(0.8)' : 'scale(1)',
    };

    const iconStyle: React.CSSProperties = {
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      fontSize: tileSize * 0.6,
      color: '#fff',
    };

    return (
      <div key={gate.id} style={style} className="gate" data-open={gate.isOpen}>
        <span style={iconStyle}>{gate.isOpen ? '⬆' : '⬇'}</span>
      </div>
    );
  };

  // Manual gate control (for testing/debugging)
  const handleGateClick = useCallback((gateId: string, isOpen: boolean) => {
    if (isOpen) {
      wallStore.closeGate(gateId);
    } else {
      wallStore.openGate(gateId);
    }
  }, []);

  return (
    <div
      className="wall-system"
      style={{
        position: 'relative',
        width: gridWidth * tileSize,
        height: gridHeight * tileSize,
        backgroundColor: '#1a202c',
      }}
    >
      {/* Render all walls */}
      {walls.map(renderWall)}

      {/* Render all gates with click handlers */}
      {gates.map((gate) => (
        <div
          key={gate.id}
          onClick={() => handleGateClick(gate.id, gate.isOpen)}
          style={{ cursor: 'pointer' }}
        >
          {renderGate(gate)}
        </div>
      ))}

      {/* Debug overlay */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: '#fff',
          padding: '10px',
          borderRadius: '4px',
          fontSize: '12px',
          pointerEvents: 'none',
        }}
      >
        <div>Walls: {walls.length}</div>
        <div>Gates: {gates.length}</div>
        <div>
          Open Gates: {gates.filter((g) => g.isOpen).length}
        </div>
      </div>
    </div>
  );
};

// Helper hook for using the wall system in other components
export const useWallSystem = () => {
  const [state, setState] = useState(wallStore.getState());

  useEffect(() => {
    const unsubscribe = wallStore.subscribe(setState);
    return unsubscribe;
  }, []);

  return {
    walls: wallStore.getWalls(),
    gates: wallStore.getGates(),
    addWall: (wall: Wall) => wallStore.addWall(wall),
    removeWall: (id: string) => wallStore.removeWall(id),
    openGate: (id: string) => wallStore.openGate(id),
    closeGate: (id: string) => wallStore.closeGate(id),
    isPositionBlocked: (pos: Position, team: 'friendly' | 'enemy') =>
      wallStore.isPositionBlocked(pos, team),
  };
};

export default WallSystem;
