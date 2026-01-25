import React from 'react';
import { Unit as UnitType } from '../types';

interface UnitProps {
  unit: UnitType;
  onSelect: (id: string) => void;
  onRightClick: (id: string) => void;
}

const Unit: React.FC<UnitProps> = ({ unit, onSelect, onRightClick }) => {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    onSelect(unit.id);
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    onRightClick(unit.id);
  };

  const unitColor = unit.team === 'player' ? '#4CAF50' : '#f44336';
  const borderColor = unit.isSelected ? '#FFD700' : unitColor;

  const healthPercentage = (unit.health / unit.maxHealth) * 100;

  const unitStyle: React.CSSProperties = {
    position: 'absolute',
    left: `${unit.position.x - 20}px`,
    top: `${unit.position.y - 20}px`,
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    backgroundColor: unitColor,
    border: `3px solid ${borderColor}`,
    cursor: 'pointer',
    boxShadow: unit.isSelected
      ? '0 0 10px 3px rgba(255, 215, 0, 0.6)'
      : '0 2px 4px rgba(0, 0, 0, 0.3)',
    transition: 'border-color 0.1s, box-shadow 0.1s',
  };

  const healthBarContainerStyle: React.CSSProperties = {
    position: 'absolute',
    bottom: '-8px',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '45px',
    height: '5px',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: '2px',
    overflow: 'hidden',
  };

  const healthBarStyle: React.CSSProperties = {
    height: '100%',
    width: `${healthPercentage}%`,
    backgroundColor: healthPercentage > 50 ? '#4CAF50' : healthPercentage > 25 ? '#FFC107' : '#f44336',
    transition: 'width 0.2s',
  };

  return (
    <div
      style={unitStyle}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      <div style={healthBarContainerStyle}>
        <div style={healthBarStyle} />
      </div>
    </div>
  );
};

export default Unit;
