import React from 'react'
import { BuildingData, getBuildingColor } from '../store/gameStore'

interface BuildingProps {
  building: BuildingData
  isGhost?: boolean
}

export const Building: React.FC<BuildingProps> = ({ building, isGhost = false }) => {
  const color = getBuildingColor(building.type)
  
  const style: React.CSSProperties = {
    position: 'absolute',
    left: `${building.x}px`,
    top: `${building.y}px`,
    width: `${building.width}px`,
    height: `${building.height}px`,
    backgroundColor: isGhost ? `${color}80` : color,
    border: isGhost ? '2px dashed #fff' : '2px solid #2c3e50',
    borderRadius: '4px',
    transform: `translate(-50%, -50%) rotate(${building.rotation}deg)`,
    transformOrigin: 'center center',
    transition: isGhost ? 'none' : 'all 0.2s ease',
    cursor: isGhost ? 'none' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'column',
    opacity: isGhost ? 0.7 : 1,
    pointerEvents: isGhost ? 'none' : 'auto',
    boxShadow: isGhost ? 'none' : '0 4px 8px rgba(0,0,0,0.3)'
  }
  
  return (
    <div style={style}>
      <div style={{
        color: '#fff',
        fontSize: '12px',
        fontWeight: 'bold',
        textTransform: 'capitalize',
        textAlign: 'center',
        userSelect: 'none',
        textShadow: '1px 1px 2px rgba(0,0,0,0.5)'
      }}>
        {building.type}
      </div>
      {isGhost && (
        <div style={{
          color: '#fff',
          fontSize: '10px',
          marginTop: '4px',
          opacity: 0.8,
          userSelect: 'none'
        }}>
          {building.rotation}°
        </div>
      )}
      {!isGhost && (
        <div style={{
          position: 'absolute',
          top: '4px',
          right: '4px',
          width: '8px',
          height: '8px',
          backgroundColor: '#27ae60',
          borderRadius: '50%',
          boxShadow: '0 0 4px #27ae60'
        }} />
      )}
    </div>
  )
}

export default Building
