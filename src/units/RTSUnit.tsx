import React from 'react';
import { Unit } from '../types';

interface RTSUnitProps {
  unit: Unit;
  onHover: (unit: Unit | null) => void;
  onClick: (unit: Unit) => void;
}

const RTSUnit: React.FC<RTSUnitProps> = ({ unit, onHover, onClick }) => {
  const handleMouseEnter = () => {
    onHover(unit);
  };

  const handleMouseLeave = () => {
    onHover(null);
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClick(unit);
  };

  const healthPercentage = (unit.health / unit.maxHealth) * 100;

  return (
    <div
      className={`unit ${unit.type} ${unit.isSelected ? 'selected' : ''}`}
      style={{
        left: `${unit.position.x}px`,
        top: `${unit.position.y}px`,
        width: '60px',
        height: '60px',
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Unit Visual */}
      <div
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: unit.type === 'friendly' ? '#4CAF50' : '#F44336',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          fontSize: '12px',
          border: unit.isSelected ? '3px solid yellow' : 'none',
        }}
      >
        {unit.type === 'friendly' ? '🛡️' : '⚔️'}
      </div>

      {/* Health Bar */}
      <div className="health-bar">
        <div
          className={`health-bar-fill ${unit.type}`}
          style={{ width: `${healthPercentage}%` }}
        />
      </div>

      {/* Unit Name (visible on hover) */}
      <div
        style={{
          position: 'absolute',
          top: '-20px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '11px',
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
        }}
      >
        {unit.name}
      </div>
    </div>
  );
};

export default RTSUnit;
