'use client'

import React from 'react'

interface GroupDestinationMarkerProps {
  x: number
  y: number
}

export const GroupDestinationMarker: React.FC<GroupDestinationMarkerProps> = ({ x, y }) => {
  return (
    <div
      className="absolute pointer-events-none animate-pulse"
      style={{
        left: `${x - 16}px`,
        top: `${y - 16}px`,
        zIndex: 5,
      }}
    >
      {/* Outer ring */}
      <div className="relative w-8 h-8">
        <div className="absolute inset-0 rounded-full border-2 border-green-400 opacity-60 animate-ping" />
        <div className="absolute inset-0 rounded-full border-2 border-green-400" />
        
        {/* Inner dot */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-2 h-2 bg-green-400 rounded-full" />
        </div>
        
        {/* Cross hairs */}
        <div className="absolute top-1/2 left-0 w-full h-0.5 bg-green-400 opacity-50" style={{ transform: 'translateY(-50%)' }} />
        <div className="absolute left-1/2 top-0 w-0.5 h-full bg-green-400 opacity-50" style={{ transform: 'translateX(-50%)' }} />
      </div>
    </div>
  )
}

export default GroupDestinationMarker
