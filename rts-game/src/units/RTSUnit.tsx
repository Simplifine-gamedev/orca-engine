import { useEffect, useRef } from 'react';
import { Unit } from '../types';
import { useGameStore } from '../store/gameStore';

interface RTSUnitProps {
  unit: Unit;
}

const UNIT_SIZE = 40;

const getUnitColor = (team: 'player' | 'enemy'): string => {
  return team === 'player' ? '#4a90e2' : '#e24a4a';
};

const getUnitTypeSymbol = (type: string): string => {
  switch (type) {
    case 'soldier': return '🎖️';
    case 'tank': return '🚜';
    case 'scout': return '👁️';
    default: return '•';
  }
};

export const RTSUnit: React.FC<RTSUnitProps> = ({ unit }) => {
  const unitRef = useRef<HTMLDivElement>(null);
  const { hoveredUnitId, setHoveredUnit } = useGameStore();
  
  const isHovered = hoveredUnitId === unit.id;
  const healthPercentage = (unit.health / unit.maxHealth) * 100;
  const isDead = unit.health <= 0;

  useEffect(() => {
    if (isDead && unitRef.current) {
      unitRef.current.style.opacity = '0.3';
    }
  }, [isDead]);

  const handleMouseEnter = () => {
    setHoveredUnit(unit.id);
  };

  const handleMouseLeave = () => {
    setHoveredUnit(null);
  };

  if (isDead) {
    return (
      <div
        ref={unitRef}
        style={{
          position: 'absolute',
          left: unit.position.x - UNIT_SIZE / 2,
          top: unit.position.y - UNIT_SIZE / 2,
          width: UNIT_SIZE,
          height: UNIT_SIZE,
          pointerEvents: 'none',
          transition: 'opacity 0.5s ease-out',
          opacity: 0.3,
        }}
      >
        <div
          style={{
            width: '100%',
            height: '100%',
            backgroundColor: '#666',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px',
          }}
        >
          ☠️
        </div>
      </div>
    );
  }

  return (
    <div
      ref={unitRef}
      data-unit-id={unit.id}
      style={{
        position: 'absolute',
        left: unit.position.x - UNIT_SIZE / 2,
        top: unit.position.y - UNIT_SIZE / 2,
        width: UNIT_SIZE,
        height: UNIT_SIZE,
        transition: 'all 0.1s ease-out',
        cursor: unit.team === 'player' ? 'pointer' : 'crosshair',
        zIndex: unit.isSelected ? 100 : 10,
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Selection ring - improved visual feedback */}
      {unit.isSelected && (
        <div
          style={{
            position: 'absolute',
            top: -6,
            left: -6,
            right: -6,
            bottom: -6,
            border: '3px solid #ffeb3b',
            borderRadius: '50%',
            animation: 'pulse 1.5s ease-in-out infinite',
            boxShadow: '0 0 10px rgba(255, 235, 59, 0.5)',
          }}
        />
      )}

      {/* Hover ring */}
      {isHovered && !unit.isSelected && (
        <div
          style={{
            position: 'absolute',
            top: -4,
            left: -4,
            right: -4,
            bottom: -4,
            border: '2px solid rgba(255, 255, 255, 0.5)',
            borderRadius: '50%',
          }}
        />
      )}

      {/* Unit body */}
      <div
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: getUnitColor(unit.team),
          border: '2px solid rgba(0, 0, 0, 0.3)',
          borderRadius: '4px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          boxShadow: unit.isSelected 
            ? '0 4px 8px rgba(0, 0, 0, 0.3)' 
            : '0 2px 4px rgba(0, 0, 0, 0.2)',
          transform: unit.isSelected ? 'scale(1.1)' : 'scale(1)',
        }}
      >
        {getUnitTypeSymbol(unit.type)}
      </div>

      {/* Health bar */}
      <div
        style={{
          position: 'absolute',
          bottom: -10,
          left: '50%',
          transform: 'translateX(-50%)',
          width: UNIT_SIZE + 4,
          height: 6,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          borderRadius: '3px',
          overflow: 'hidden',
          border: '1px solid rgba(0, 0, 0, 0.3)',
        }}
      >
        <div
          style={{
            width: `${healthPercentage}%`,
            height: '100%',
            backgroundColor: 
              healthPercentage > 60 
                ? '#4caf50' 
                : healthPercentage > 30 
                  ? '#ff9800' 
                  : '#f44336',
            transition: 'width 0.3s ease-out, background-color 0.3s ease-out',
          }}
        />
      </div>

      {/* Movement path indicator */}
      {unit.targetPosition && (
        <svg
          style={{
            position: 'absolute',
            top: UNIT_SIZE / 2,
            left: UNIT_SIZE / 2,
            pointerEvents: 'none',
            overflow: 'visible',
            zIndex: 5,
          }}
        >
          <line
            x1={0}
            y1={0}
            x2={unit.targetPosition.x - unit.position.x}
            y2={unit.targetPosition.y - unit.position.y}
            stroke="rgba(255, 255, 255, 0.4)"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
        </svg>
      )}

      {/* Team indicator badge */}
      <div
        style={{
          position: 'absolute',
          top: -8,
          right: -8,
          width: 16,
          height: 16,
          borderRadius: '50%',
          backgroundColor: unit.team === 'player' ? '#4caf50' : '#f44336',
          border: '2px solid white',
          fontSize: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          color: 'white',
        }}
      >
        {unit.team === 'player' ? 'P' : 'E'}
      </div>
    </div>
  );
};

// Add keyframe animation for pulse effect
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.7;
      transform: scale(1.05);
    }
  }
`;
document.head.appendChild(style);
