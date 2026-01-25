import React, { useEffect } from 'react';
import { useGameStore } from '../store/gameStore';
import { RTSUnit } from '../units/RTSUnit';

/**
 * Example implementation showing how to use the RTS unit system
 * with proper animation state management to prevent T-posing
 */
export const GameExample: React.FC = () => {
  const { units, buildings, spawnUnit } = useGameStore();

  // Initialize a sample building
  useEffect(() => {
    // In a real game, buildings would be created through game logic
    useGameStore.setState({
      buildings: new Map([
        [
          'barracks-1',
          {
            id: 'barracks-1',
            type: 'barracks',
            position: { x: 200, y: 200 },
            spawnPoint: { x: 250, y: 250 }, // Where units appear
          },
        ],
      ]),
    });
  }, []);

  const handleSpawnSoldier = () => {
    // This will spawn a unit with proper animation state
    // Unit will show 'spawning' animation then transition to 'idle'
    spawnUnit('barracks-1', 'soldier');
  };

  const handleSpawnArcher = () => {
    spawnUnit('barracks-1', 'archer');
  };

  return (
    <div className="game-container" style={{ position: 'relative', width: '100%', height: '600px' }}>
      <div className="game-viewport">
        {/* Render all units with proper animation handling */}
        {Array.from(units.values()).map((unit) => (
          <RTSUnit
            key={unit.id}
            unit={unit}
            onAnimationComplete={(state) => {
              console.log(`Unit ${unit.id} completed ${state} animation`);
            }}
          />
        ))}
      </div>

      <div className="controls" style={{ position: 'absolute', top: 10, left: 10 }}>
        <button onClick={handleSpawnSoldier}>Spawn Soldier</button>
        <button onClick={handleSpawnArcher}>Spawn Archer</button>
        <div className="info">
          Units: {units.size}
        </div>
      </div>

      <style>{`
        .game-container {
          background: #1a1a1a;
          border: 2px solid #333;
        }

        .controls button {
          margin: 5px;
          padding: 10px 20px;
          background: #4CAF50;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }

        .controls button:hover {
          background: #45a049;
        }

        .info {
          margin-top: 10px;
          color: white;
          font-family: monospace;
        }

        .rts-unit {
          width: 40px;
          height: 40px;
        }

        .unit-sprite {
          width: 100%;
          height: 100%;
          background: #3498db;
          border: 2px solid #2980b9;
          border-radius: 4px;
          position: relative;
        }

        .health-bar {
          position: absolute;
          top: -8px;
          left: 0;
          width: 100%;
          height: 4px;
          background: #333;
          border-radius: 2px;
        }

        .health-fill {
          height: 100%;
          background: #4CAF50;
          border-radius: 2px;
          transition: width 0.3s;
        }

        .animation-state {
          position: absolute;
          bottom: -16px;
          left: 50%;
          transform: translateX(-50%);
          font-size: 8px;
          color: white;
          white-space: nowrap;
        }
      `}</style>
    </div>
  );
};

export default GameExample;
