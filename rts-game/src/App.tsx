import React, { useState } from 'react';
import ControlPoint from './objects/ControlPoint';
import { ControlPointData } from './types/ControlPoint';

const PLAYER_ID = 'player-1';
const ENEMY_ID = 'enemy-1';

const App: React.FC = () => {
  const [controlPoints, setControlPoints] = useState<ControlPointData[]>([
    {
      id: 'cp-1',
      position: { x: 100, y: 0, z: 100 },
      ownerId: null,
      captureProgress: 0,
      captureRadius: 50,
      name: 'North Point'
    },
    {
      id: 'cp-2',
      position: { x: 300, y: 0, z: 100 },
      ownerId: PLAYER_ID,
      captureProgress: 100,
      captureRadius: 50,
      name: 'Center Point'
    },
    {
      id: 'cp-3',
      position: { x: 500, y: 0, z: 100 },
      ownerId: ENEMY_ID,
      captureProgress: 100,
      captureRadius: 50,
      name: 'South Point'
    }
  ]);

  const capturePoint = (pointId: string) => {
    setControlPoints(prev => prev.map(point => 
      point.id === pointId 
        ? { ...point, ownerId: PLAYER_ID, captureProgress: 100 }
        : point
    ));
  };

  return (
    <div style={{ 
      width: '100%', 
      height: '100vh', 
      backgroundColor: '#1a1a1a',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <div style={{ 
        padding: '20px', 
        color: 'white',
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: 10,
        backgroundColor: 'rgba(0,0,0,0.7)',
        borderRadius: '8px',
        margin: '10px'
      }}>
        <h1>Orca RTS - Control Points Demo</h1>
        <p>Click on neutral or enemy control points to capture them</p>
        <div style={{ marginTop: '10px' }}>
          <div>🟢 <strong>Controlled</strong> - Points you own</div>
          <div>🔴 <strong>Enemy</strong> - Points owned by enemies</div>
          <div>⚪ <strong>Neutral</strong> - Unclaimed points</div>
        </div>
      </div>

      {controlPoints.map(point => (
        <div 
          key={point.id} 
          onClick={() => point.ownerId !== PLAYER_ID && capturePoint(point.id)}
          style={{ cursor: point.ownerId !== PLAYER_ID ? 'pointer' : 'default' }}
        >
          <ControlPoint 
            point={point} 
            playerId={PLAYER_ID} 
            enemyId={ENEMY_ID} 
          />
        </div>
      ))}
    </div>
  );
};

export default App;
