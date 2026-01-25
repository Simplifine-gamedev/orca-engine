import React from 'react';
import { Unit } from '../types/unit';

interface RTSUnitProps {
  unit: Unit;
  isSelected: boolean;
  onClick: (e: React.MouseEvent) => void;
}

const RTSUnit: React.FC<RTSUnitProps> = ({ unit, isSelected, onClick }) => {
  // Unit colors based on type
  const unitColors = {
    warrior: '#FF6B6B',
    archer: '#4ECDC4',
    mage: '#A78BFA'
  };
  
  // Selection indicator styles
  const selectionRingStyle: React.CSSProperties = {
    position: 'absolute',
    left: unit.position.x - 35,
    top: unit.position.y - 35,
    width: 70,
    height: 70,
    border: '3px solid #FFD700',
    borderRadius: '50%',
    boxShadow: '0 0 20px rgba(255, 215, 0, 0.6)',
    pointerEvents: 'none',
    animation: 'pulse 1.5s ease-in-out infinite',
    zIndex: 1
  };
  
  // Unit body style
  const unitStyle: React.CSSProperties = {
    position: 'absolute',
    left: unit.position.x - 25,
    top: unit.position.y - 25,
    width: 50,
    height: 50,
    backgroundColor: unitColors[unit.type],
    border: isSelected ? '3px solid #FFD700' : '2px solid #333',
    borderRadius: '50%',
    cursor: 'pointer',
    transition: 'transform 0.1s ease, border 0.1s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: isSelected 
      ? '0 0 15px rgba(255, 215, 0, 0.8)' 
      : '0 2px 4px rgba(0,0,0,0.2)',
    zIndex: 2
  };
  
  // Health bar style
  const healthBarContainerStyle: React.CSSProperties = {
    position: 'absolute',
    left: unit.position.x - 25,
    top: unit.position.y - 35,
    width: 50,
    height: 6,
    backgroundColor: 'rgba(0,0,0,0.3)',
    borderRadius: 3,
    overflow: 'hidden',
    border: '1px solid rgba(255,255,255,0.3)',
    zIndex: 3
  };
  
  const healthBarStyle: React.CSSProperties = {
    height: '100%',
    width: `${(unit.health / unit.maxHealth) * 100}%`,
    backgroundColor: unit.health > unit.maxHealth * 0.5 ? '#4ADE80' : 
                     unit.health > unit.maxHealth * 0.25 ? '#FBBF24' : '#EF4444',
    transition: 'width 0.3s ease, background-color 0.3s ease',
    boxShadow: '0 0 5px rgba(0,0,0,0.3)'
  };
  
  // Movement indicator (target position)
  const targetIndicatorStyle: React.CSSProperties = unit.targetPosition ? {
    position: 'absolute',
    left: unit.targetPosition.x - 8,
    top: unit.targetPosition.y - 8,
    width: 16,
    height: 16,
    border: '2px solid #FFD700',
    borderRadius: '50%',
    backgroundColor: 'rgba(255, 215, 0, 0.3)',
    pointerEvents: 'none',
    animation: 'blink 0.8s ease-in-out infinite',
    zIndex: 1
  } : {};
  
  // Movement line from unit to target
  const renderMovementLine = () => {
    if (!unit.targetPosition || !unit.isMoving) return null;
    
    const dx = unit.targetPosition.x - unit.position.x;
    const dy = unit.targetPosition.y - unit.position.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    
    const lineStyle: React.CSSProperties = {
      position: 'absolute',
      left: unit.position.x,
      top: unit.position.y,
      width: length,
      height: 2,
      backgroundColor: 'rgba(255, 215, 0, 0.5)',
      transformOrigin: '0 50%',
      transform: `rotate(${angle}deg)`,
      pointerEvents: 'none',
      zIndex: 0
    };
    
    return <div style={lineStyle} />;
  };
  
  // Unit type icon
  const getUnitIcon = () => {
    const icons = {
      warrior: '⚔️',
      archer: '🏹',
      mage: '🔮'
    };
    return icons[unit.type];
  };
  
  return (
    <>
      {/* Movement line */}
      {renderMovementLine()}
      
      {/* Target position indicator */}
      {unit.targetPosition && unit.isMoving && (
        <div style={targetIndicatorStyle} />
      )}
      
      {/* Selection ring */}
      {isSelected && <div style={selectionRingStyle} />}
      
      {/* Health bar */}
      <div style={healthBarContainerStyle}>
        <div style={healthBarStyle} />
      </div>
      
      {/* Unit body */}
      <div 
        style={unitStyle}
        onClick={onClick}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.transform = 'scale(1.1)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.transform = 'scale(1)';
        }}
      >
        <span style={{ fontSize: 24 }}>{getUnitIcon()}</span>
      </div>
      
      {/* CSS animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.05); }
        }
        
        @keyframes blink {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
      `}</style>
    </>
  );
};

export default RTSUnit;
