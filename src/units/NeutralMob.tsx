import React from 'react';
import { Unit } from '../types';

interface NeutralMobProps {
  unit: Unit;
  onHover: (unit: Unit | null) => void;
  onClick: (unit: Unit) => void;
}

const NeutralMob: React.FC<NeutralMobProps> = ({ unit, onHover, onClick }) => {
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
        width: '70px',
        height: '70px',
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Mob Visual - Neutral mobs are typically larger and have distinct appearance */}
      <div
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#FF9800',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          fontSize: '28px',
          border: unit.isSelected ? '3px solid yellow' : '2px solid #E65100',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
        }}
      >
        🐺
      </div>

      {/* Health Bar */}
      <div className="health-bar">
        <div
          className={`health-bar-fill ${unit.type}`}
          style={{ width: `${healthPercentage}%` }}
        />
      </div>

      {/* Mob Name (visible on hover) */}
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

      {/* Level indicator for neutral mobs */}
      <div
        style={{
          position: 'absolute',
          bottom: '-20px',
          right: '0',
          backgroundColor: 'rgba(255, 152, 0, 0.9)',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '10px',
          fontWeight: 'bold',
        }}
      >
        Lv.{Math.floor(unit.maxHealth / 20)}
      </div>
    </div>
  );
};

export default NeutralMob;
