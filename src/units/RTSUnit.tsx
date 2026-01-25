import React from 'react';
import { UnitType } from '../App';

interface RTSUnitProps {
  id: string;
  x: number;
  y: number;
  type: UnitType;
  health: number;
  maxHealth: number;
  isHovered: boolean;
  isSelected: boolean;
  onHover: (unitId: string | null, unitType: UnitType | null) => void;
  onClick: (unitId: string, unitType: UnitType) => void;
}

const RTSUnit: React.FC<RTSUnitProps> = ({
  id,
  x,
  y,
  type,
  health,
  maxHealth,
  isHovered,
  isSelected,
  onHover,
  onClick,
}) => {
  const handleMouseEnter = () => {
    onHover(id, type);
  };

  const handleMouseLeave = () => {
    onHover(null, null);
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClick(id, type);
  };

  const healthPercent = (health / maxHealth) * 100;
  
  // Determine unit color based on type
  const getUnitColor = () => {
    switch (type) {
      case 'player':
        return '#3b82f6'; // blue
      case 'enemy':
        return '#ef4444'; // red
      default:
        return '#64748b'; // gray
    }
  };

  const unitColor = getUnitColor();

  return (
    <div
      className={`rts-unit ${type} ${isHovered ? 'hovered' : ''} ${isSelected ? 'selected' : ''}`}
      style={{
        left: x,
        top: y,
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Unit circle */}
      <div 
        className="unit-body"
        style={{
          backgroundColor: unitColor,
          boxShadow: isHovered 
            ? `0 0 20px 5px ${unitColor}aa` 
            : isSelected 
              ? `0 0 15px 3px ${unitColor}` 
              : 'none',
        }}
      />
      
      {/* Hover highlight ring */}
      {isHovered && (
        <div 
          className="hover-ring"
          style={{
            borderColor: unitColor,
          }}
        />
      )}
      
      {/* Selection indicator */}
      {isSelected && (
        <div className="selection-ring" />
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
      
      {/* Unit label */}
      <div className="unit-label">
        {type === 'player' ? '🛡️' : '⚔️'}
      </div>
    </div>
  );
};

export default RTSUnit;
