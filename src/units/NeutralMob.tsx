import React from 'react';

interface NeutralMobProps {
  id: string;
  x: number;
  y: number;
  health: number;
  maxHealth: number;
  isHovered: boolean;
  onHover: (unitId: string | null, unitType: 'neutral' | null) => void;
  onClick: (unitId: string, unitType: 'neutral') => void;
}

const NeutralMob: React.FC<NeutralMobProps> = ({
  id,
  x,
  y,
  health,
  maxHealth,
  isHovered,
  onHover,
  onClick,
}) => {
  const handleMouseEnter = () => {
    onHover(id, 'neutral');
  };

  const handleMouseLeave = () => {
    onHover(null, null);
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClick(id, 'neutral');
  };

  const healthPercent = (health / maxHealth) * 100;
  const mobColor = '#f59e0b'; // amber/yellow for neutral

  return (
    <div
      className={`neutral-mob ${isHovered ? 'hovered' : ''}`}
      style={{
        left: x,
        top: y,
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Mob body - slightly different shape from regular units */}
      <div 
        className="mob-body"
        style={{
          backgroundColor: mobColor,
          boxShadow: isHovered 
            ? `0 0 25px 8px ${mobColor}99` 
            : 'none',
        }}
      />
      
      {/* Hover highlight ring with pulsing effect */}
      {isHovered && (
        <>
          <div 
            className="hover-ring pulse"
            style={{
              borderColor: mobColor,
            }}
          />
          <div 
            className="hover-ring-outer pulse-slow"
            style={{
              borderColor: mobColor,
            }}
          />
        </>
      )}
      
      {/* Health bar */}
      <div className="health-bar-container">
        <div 
          className="health-bar"
          style={{
            width: `${healthPercent}%`,
            backgroundColor: healthPercent > 50 ? '#22c55e' : healthPercent > 25 ? '#eab308' : '#ef4444',
          }}
        />
      </div>
      
      {/* Mob label/icon */}
      <div className="unit-label">
        🐲
      </div>
      
      {/* Attackable indicator that appears on hover */}
      {isHovered && (
        <div className="attackable-indicator">
          ⚔️
        </div>
      )}
    </div>
  );
};

export default NeutralMob;
