'use client'

import React, { useEffect, useRef } from 'react'
import useGameStore, { Unit } from '../store/gameStore'

interface RTSUnitProps {
  unit: Unit
}

export const RTSUnit: React.FC<RTSUnitProps> = ({ unit }) => {
  const pathSettings = useGameStore((state) => state.pathSettings)
  const pathRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    // Fade out path lines based on fadeSpeed
    if (pathRef.current && unit.targetX !== null) {
      const fadeTimeout = setTimeout(() => {
        if (pathRef.current) {
          pathRef.current.style.transition = `opacity ${pathSettings.pathFadeSpeed}s ease-out`
          pathRef.current.style.opacity = '0'
        }
      }, 100)
      
      return () => clearTimeout(fadeTimeout)
    }
  }, [unit.targetX, unit.targetY, pathSettings.pathFadeSpeed])
  
  const shouldShowPath = () => {
    // Don't show if paths are disabled globally
    if (!pathSettings.showPaths) return false
    
    // Don't show if no target
    if (unit.targetX === null || unit.targetY === null) return false
    
    // If only lead unit should show path
    if (pathSettings.showOnlyLeadUnit && !unit.isLeader) return false
    
    return true
  }
  
  const renderPathLine = () => {
    if (!shouldShowPath()) return null
    
    const dx = unit.targetX! - unit.x
    const dy = unit.targetY! - unit.y
    const length = Math.sqrt(dx * dx + dy * dy)
    const angle = Math.atan2(dy, dx) * (180 / Math.PI)
    
    return (
      <div
        ref={pathRef}
        className="absolute pointer-events-none"
        style={{
          left: `${unit.x}px`,
          top: `${unit.y}px`,
          width: `${length}px`,
          height: '2px',
          backgroundColor: unit.isLeader ? '#3b82f6' : '#60a5fa',
          transformOrigin: '0 0',
          transform: `rotate(${angle}deg)`,
          opacity: pathSettings.pathOpacity,
          transition: `opacity ${pathSettings.pathFadeSpeed}s ease-out`,
          zIndex: 1,
        }}
      >
        {/* Arrow head */}
        <div
          className="absolute"
          style={{
            right: '-6px',
            top: '-3px',
            width: '0',
            height: '0',
            borderLeft: '6px solid currentColor',
            borderTop: '4px solid transparent',
            borderBottom: '4px solid transparent',
            color: unit.isLeader ? '#3b82f6' : '#60a5fa',
          }}
        />
      </div>
    )
  }
  
  return (
    <>
      {renderPathLine()}
      
      {/* Unit circle */}
      <div
        className="absolute rounded-full transition-all duration-200 cursor-pointer"
        style={{
          left: `${unit.x - 12}px`,
          top: `${unit.y - 12}px`,
          width: '24px',
          height: '24px',
          backgroundColor: unit.isSelected ? '#3b82f6' : '#6b7280',
          border: unit.isLeader ? '3px solid #fbbf24' : unit.isSelected ? '2px solid #60a5fa' : '2px solid #4b5563',
          boxShadow: unit.isSelected ? '0 0 12px rgba(59, 130, 246, 0.6)' : 'none',
          zIndex: 10,
        }}
      >
        {/* Leader indicator */}
        {unit.isLeader && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-pulse" />
        )}
      </div>
    </>
  )
}

export default RTSUnit
