import React from 'react';
import { PlacedBuilding, BuildingType } from '../store/gameStore';

interface BuildingProps {
  building?: PlacedBuilding;
  buildingType?: BuildingType;
  x: number;
  y: number;
  rotation: number;
  isGhost?: boolean;
  onClick?: () => void;
}

const GRID_SIZE = 32; // pixels per grid cell

/**
 * Building component that displays a building on the game map.
 * Supports rotation in 90-degree increments (0, 90, 180, 270).
 */
export const Building: React.FC<BuildingProps> = ({
  building,
  buildingType,
  x,
  y,
  rotation,
  isGhost = false,
  onClick,
}) => {
  const type = building?.type || buildingType;
  
  if (!type) return null;
  
  // Calculate dimensions based on rotation
  // When rotated 90 or 270 degrees, width and height are swapped
  const isRotated90or270 = rotation === 90 || rotation === 270;
  const displayWidth = isRotated90or270 ? type.height : type.width;
  const displayHeight = isRotated90or270 ? type.width : type.height;
  
  const style: React.CSSProperties = {
    position: 'absolute',
    left: x * GRID_SIZE,
    top: y * GRID_SIZE,
    width: displayWidth * GRID_SIZE,
    height: displayHeight * GRID_SIZE,
    backgroundColor: type.color,
    border: isGhost ? '2px dashed #fff' : '2px solid #000',
    opacity: isGhost ? 0.5 : 1,
    cursor: onClick ? 'pointer' : 'default',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#fff',
    textShadow: '1px 1px 2px #000',
    transition: 'all 0.1s ease',
    boxSizing: 'border-box',
    transform: `rotate(${rotation}deg)`,
    transformOrigin: 'center center',
  };
  
  return (
    <div 
      style={style} 
      onClick={onClick}
      title={`${type.name} (${rotation}°)`}
    >
      <div style={{ transform: `rotate(${-rotation}deg)` }}>
        {type.name}
        {rotation !== 0 && (
          <div style={{ fontSize: '10px', marginTop: '4px' }}>
            {rotation}°
          </div>
        )}
      </div>
    </div>
  );
};

export default Building;
