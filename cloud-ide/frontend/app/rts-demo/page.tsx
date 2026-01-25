'use client'

import React, { useEffect, useRef } from 'react'
import useGameStore from '../../src/store/gameStore'
import RTSUnit from '../../src/units/RTSUnit'
import GroupDestinationMarker from '../../src/components/GroupDestinationMarker'

export default function RTSDemo() {
  const {
    units,
    selectedUnits,
    pathSettings,
    addUnit,
    selectUnit,
    selectUnits,
    moveUnits,
    clearSelection,
    togglePathVisibility,
    toggleLeadUnitOnly,
    toggleGroupDestination,
    setPathOpacity,
    setPathFadeSpeed,
  } = useGameStore()
  
  const gameAreaRef = useRef<HTMLDivElement>(null)
  const selectionBoxRef = useRef<HTMLDivElement>(null)
  const selectionStart = useRef<{ x: number; y: number } | null>(null)
  
  // Initialize some units
  useEffect(() => {
    if (units.length === 0) {
      for (let i = 0; i < 8; i++) {
        addUnit(150 + i * 40, 200 + (i % 3) * 40)
      }
    }
  }, [])
  
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target !== gameAreaRef.current) return
    
    const rect = gameAreaRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    if (e.shiftKey) {
      // Add new unit
      addUnit(x, y)
    } else if (selectedUnits.length > 0) {
      // Move selected units
      moveUnits(x, y)
    } else {
      clearSelection()
    }
  }
  
  const handleUnitClick = (e: React.MouseEvent, unitId: string) => {
    e.stopPropagation()
    selectUnit(unitId, e.ctrlKey || e.metaKey)
  }
  
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target !== gameAreaRef.current) return
    
    const rect = gameAreaRef.current!.getBoundingClientRect()
    selectionStart.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    }
    
    if (selectionBoxRef.current) {
      selectionBoxRef.current.style.display = 'block'
      selectionBoxRef.current.style.left = `${selectionStart.current.x}px`
      selectionBoxRef.current.style.top = `${selectionStart.current.y}px`
      selectionBoxRef.current.style.width = '0'
      selectionBoxRef.current.style.height = '0'
    }
  }
  
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!selectionStart.current || !selectionBoxRef.current) return
    
    const rect = gameAreaRef.current!.getBoundingClientRect()
    const currentX = e.clientX - rect.left
    const currentY = e.clientY - rect.top
    
    const width = Math.abs(currentX - selectionStart.current.x)
    const height = Math.abs(currentY - selectionStart.current.y)
    const left = Math.min(currentX, selectionStart.current.x)
    const top = Math.min(currentY, selectionStart.current.y)
    
    selectionBoxRef.current.style.left = `${left}px`
    selectionBoxRef.current.style.top = `${top}px`
    selectionBoxRef.current.style.width = `${width}px`
    selectionBoxRef.current.style.height = `${height}px`
  }
  
  const handleMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!selectionStart.current || !selectionBoxRef.current) return
    
    const rect = gameAreaRef.current!.getBoundingClientRect()
    const endX = e.clientX - rect.left
    const endY = e.clientY - rect.top
    
    const left = Math.min(endX, selectionStart.current.x)
    const right = Math.max(endX, selectionStart.current.x)
    const top = Math.min(endY, selectionStart.current.y)
    const bottom = Math.max(endY, selectionStart.current.y)
    
    // Select units within the box
    const selectedIds = units
      .filter((unit) => unit.x >= left && unit.x <= right && unit.y >= top && unit.y <= bottom)
      .map((unit) => unit.id)
    
    if (selectedIds.length > 0) {
      selectUnits(selectedIds)
    }
    
    selectionBoxRef.current.style.display = 'none'
    selectionStart.current = null
  }
  
  // Calculate group destination for marker
  const groupDestination = selectedUnits.length > 0
    ? units.find((u) => selectedUnits.includes(u.id) && u.targetX !== null)
    : null
  
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Settings Panel */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 p-6 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-2">RTS Path Demo</h1>
        <p className="text-sm text-gray-400 mb-6">
          Demo showcasing path visibility controls for RTS units
        </p>
        
        <div className="space-y-6">
          {/* Instructions */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="font-semibold mb-2">Controls</h3>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Click unit to select</li>
              <li>• Ctrl/Cmd + Click for multi-select</li>
              <li>• Drag to box select</li>
              <li>• Click canvas to move selected</li>
              <li>• Shift + Click to add unit</li>
            </ul>
          </div>
          
          {/* Path Visibility Settings */}
          <div className="space-y-4">
            <h3 className="font-semibold text-lg">Path Visibility</h3>
            
            <div className="space-y-3">
              <label className="flex items-center justify-between cursor-pointer">
                <span>Show Path Lines</span>
                <input
                  type="checkbox"
                  checked={pathSettings.showPaths}
                  onChange={togglePathVisibility}
                  className="w-5 h-5"
                />
              </label>
              
              <label className="flex items-center justify-between cursor-pointer">
                <span>Only Lead Unit Path</span>
                <input
                  type="checkbox"
                  checked={pathSettings.showOnlyLeadUnit}
                  onChange={toggleLeadUnitOnly}
                  disabled={!pathSettings.showPaths}
                  className="w-5 h-5 disabled:opacity-50"
                />
              </label>
              
              <label className="flex items-center justify-between cursor-pointer">
                <span>Group Destination Marker</span>
                <input
                  type="checkbox"
                  checked={pathSettings.showGroupDestination}
                  onChange={toggleGroupDestination}
                  className="w-5 h-5"
                />
              </label>
            </div>
          </div>
          
          {/* Path Opacity */}
          <div className="space-y-2">
            <label className="block">
              <span className="font-semibold">Path Opacity</span>
              <span className="text-sm text-gray-400 ml-2">
                {Math.round(pathSettings.pathOpacity * 100)}%
              </span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={pathSettings.pathOpacity}
              onChange={(e) => setPathOpacity(parseFloat(e.target.value))}
              disabled={!pathSettings.showPaths}
              className="w-full disabled:opacity-50"
            />
          </div>
          
          {/* Fade Speed */}
          <div className="space-y-2">
            <label className="block">
              <span className="font-semibold">Fade Speed</span>
              <span className="text-sm text-gray-400 ml-2">
                {pathSettings.pathFadeSpeed}s
              </span>
            </label>
            <input
              type="range"
              min="0.1"
              max="5"
              step="0.1"
              value={pathSettings.pathFadeSpeed}
              onChange={(e) => setPathFadeSpeed(parseFloat(e.target.value))}
              disabled={!pathSettings.showPaths}
              className="w-full disabled:opacity-50"
            />
          </div>
          
          {/* Stats */}
          <div className="bg-gray-700 rounded-lg p-4 space-y-2">
            <h3 className="font-semibold mb-2">Stats</h3>
            <div className="text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Total Units:</span>
                <span className="font-mono">{units.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Selected:</span>
                <span className="font-mono">{selectedUnits.length}</span>
              </div>
            </div>
          </div>
          
          {/* Quick Actions */}
          <div className="space-y-2">
            <button
              onClick={clearSelection}
              className="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded transition-colors"
            >
              Clear Selection
            </button>
          </div>
        </div>
      </div>
      
      {/* Game Area */}
      <div className="flex-1 relative bg-gray-950">
        <div
          ref={gameAreaRef}
          className="w-full h-full relative overflow-hidden cursor-crosshair"
          onClick={handleCanvasClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {/* Background grid */}
          <div
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage: 'linear-gradient(#4b5563 1px, transparent 1px), linear-gradient(90deg, #4b5563 1px, transparent 1px)',
              backgroundSize: '50px 50px',
            }}
          />
          
          {/* Units */}
          {units.map((unit) => (
            <div key={unit.id} onClick={(e) => handleUnitClick(e, unit.id)}>
              <RTSUnit unit={unit} />
            </div>
          ))}
          
          {/* Group Destination Marker */}
          {pathSettings.showGroupDestination &&
            groupDestination &&
            groupDestination.targetX !== null &&
            groupDestination.targetY !== null && (
              <GroupDestinationMarker
                x={groupDestination.targetX}
                y={groupDestination.targetY}
              />
            )}
          
          {/* Selection Box */}
          <div
            ref={selectionBoxRef}
            className="absolute border-2 border-blue-400 bg-blue-400 bg-opacity-10 pointer-events-none"
            style={{ display: 'none' }}
          />
          
          {/* Hint Text */}
          {units.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <div className="text-xl mb-2">No units yet</div>
                <div className="text-sm">Shift + Click to add units</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
