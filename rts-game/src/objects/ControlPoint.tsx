import React from 'react';

interface ControlPointProps {
  point: {
    id: string;
    position: { x: number; y: number; z: number };
    ownerId: string | null;
    captureProgress: number;
  };
  playerId: string;
  enemyId: string;
}

export const ControlPoint: React.FC<ControlPointProps> = ({ point, playerId, enemyId }) => {
  // Determine the status and color based on ownership
  const getOwnershipStatus = () => {
    if (point.ownerId === null) {
      return { status: 'neutral', color: '#808080' };
    } else if (point.ownerId === playerId) {
      return { status: 'controlled', color: '#00FF00' };
    } else if (point.ownerId === enemyId) {
      return { status: 'enemy', color: '#FF0000' };
    } else {
      // Handle other players in multiplayer scenarios
      return { status: 'other', color: '#FFFF00' };
    }
  };

  const { status, color } = getOwnershipStatus();

  return (
    <div 
      style={{
        position: 'absolute',
        left: point.position.x,
        top: point.position.z,
        width: '50px',
        height: '50px',
        borderRadius: '50%',
        backgroundColor: color,
        border: '3px solid white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontWeight: 'bold',
        fontSize: '12px',
        textTransform: 'uppercase',
        boxShadow: '0 0 10px rgba(0, 0, 0, 0.5)'
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div>{status}</div>
        {point.captureProgress > 0 && point.captureProgress < 100 && (
          <div style={{ fontSize: '10px' }}>{point.captureProgress}%</div>
        )}
      </div>
    </div>
  );
};

export default ControlPoint;
