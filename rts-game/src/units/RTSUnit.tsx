import React from 'react'
import { Unit } from '../types'

interface RTSUnitProps {
  unit: Unit
  showPath: boolean
  onClick: (e: React.MouseEvent) => void
}

export const RTSUnit: React.FC<RTSUnitProps> = ({ unit, showPath, onClick }) => {
  const { position, targetPosition, isSelected, color, facing } = unit
  
  return (
    <g>
      {/* Path line */}
      {showPath && targetPosition && (
        <line
          x1={position.x}
          y1={position.y}
          x2={targetPosition.x}
          y2={targetPosition.y}
          stroke={color}
          strokeWidth="2"
          strokeDasharray="5,5"
          opacity="0.6"
        />
      )}
      
      {/* Target position marker */}
      {showPath && targetPosition && (
        <circle
          cx={targetPosition.x}
          cy={targetPosition.y}
          r="4"
          fill={color}
          opacity="0.5"
        />
      )}
      
      {/* Unit circle */}
      <circle
        cx={position.x}
        cy={position.y}
        r="12"
        fill={color}
        stroke={isSelected ? '#fff' : '#000'}
        strokeWidth={isSelected ? 3 : 1}
        onClick={onClick}
        style={{ cursor: 'pointer' }}
      />
      
      {/* Facing direction indicator */}
      <line
        x1={position.x}
        y1={position.y}
        x2={position.x + Math.cos(facing) * 18}
        y2={position.y + Math.sin(facing) * 18}
        stroke={isSelected ? '#fff' : '#000'}
        strokeWidth="2"
      />
      
      {/* Selection ring */}
      {isSelected && (
        <circle
          cx={position.x}
          cy={position.y}
          r="20"
          fill="none"
          stroke="#fff"
          strokeWidth="2"
          opacity="0.5"
        />
      )}
    </g>
  )
}
