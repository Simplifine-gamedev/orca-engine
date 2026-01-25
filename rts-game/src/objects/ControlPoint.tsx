import React, { useState } from 'react';
import './ControlPoint.css';

interface ControlPointProps {
  id: string;
  x: number;
  y: number;
  team?: 'neutral' | 'player' | 'enemy';
  visionRadius?: number;
  isWatchtower?: boolean;
}

export const ControlPoint: React.FC<ControlPointProps> = ({
  id,
  x,
  y,
  team = 'neutral',
  visionRadius = 150,
  isWatchtower = true,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  const getTeamColor = () => {
    switch (team) {
      case 'player':
        return '#4CAF50';
      case 'enemy':
        return '#f44336';
      default:
        return '#9E9E9E';
    }
  };

  return (
    <div
      className="control-point"
      style={{
        left: `${x}px`,
        top: `${y}px`,
        position: 'absolute',
      }}
      onMouseEnter={() => {
        setIsHovered(true);
        setShowTooltip(true);
      }}
      onMouseLeave={() => {
        setIsHovered(false);
        setShowTooltip(false);
      }}
    >
      {/* Vision radius preview on hover */}
      {isHovered && isWatchtower && (
        <div
          className="vision-radius"
          style={{
            width: `${visionRadius * 2}px`,
            height: `${visionRadius * 2}px`,
            border: `2px dashed ${getTeamColor()}`,
            borderRadius: '50%',
            position: 'absolute',
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: `${getTeamColor()}22`,
            pointerEvents: 'none',
            animation: 'pulse 2s ease-in-out infinite',
          }}
        />
      )}

      {/* Watchtower structure */}
      <div
        className="watchtower-base"
        style={{
          width: '60px',
          height: '80px',
          backgroundColor: getTeamColor(),
          border: `3px solid ${getTeamColor()}`,
          borderRadius: '8px',
          position: 'relative',
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
        }}
      >
        {/* Tower top */}
        <div
          className="watchtower-top"
          style={{
            width: '100%',
            height: '20px',
            backgroundColor: getTeamColor(),
            borderRadius: '4px 4px 0 0',
            position: 'absolute',
            top: '-10px',
            left: '0',
          }}
        />

        {/* Eye icon indicator */}
        {isWatchtower && (
          <div
            className="eye-icon"
            style={{
              position: 'absolute',
              top: '-45px',
              left: '50%',
              transform: 'translateX(-50%)',
              fontSize: '32px',
              animation: 'float 3s ease-in-out infinite',
              filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5))',
            }}
          >
            👁️
          </div>
        )}

        {/* Control point label */}
        <div
          className="control-point-label"
          style={{
            position: 'absolute',
            bottom: '-25px',
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: '12px',
            fontWeight: 'bold',
            color: '#fff',
            textShadow: '0 1px 2px rgba(0, 0, 0, 0.8)',
            whiteSpace: 'nowrap',
          }}
        >
          {team === 'neutral' ? 'Capture' : team === 'player' ? 'Controlled' : 'Enemy'}
        </div>
      </div>

      {/* Tooltip explaining vision benefit */}
      {showTooltip && (
        <div
          className="tooltip"
          style={{
            position: 'absolute',
            top: '-100px',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: '#fff',
            padding: '12px 16px',
            borderRadius: '8px',
            fontSize: '14px',
            width: '250px',
            textAlign: 'center',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
            zIndex: 1000,
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '6px' }}>
            🗼 Watchtower
          </div>
          <div style={{ fontSize: '12px', lineHeight: '1.5' }}>
            Provides vision in a {visionRadius}px radius
            <br />
            {team === 'neutral'
              ? 'Capture to reveal enemy movements'
              : team === 'player'
              ? 'Currently providing vision for your team'
              : 'Captured by enemy'}
          </div>
        </div>
      )}
    </div>
  );
};

export default ControlPoint;
