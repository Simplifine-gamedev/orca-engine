import React, { useEffect, useRef, useState } from 'react'
import { useGameStore } from './store/gameStore'
import { RTSUnit } from './units/RTSUnit'
import './App.css'

const CANVAS_WIDTH = 1200
const CANVAS_HEIGHT = 800

function App() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [isRightMouseDown, setIsRightMouseDown] = useState(false)
  
  const {
    units,
    selectedUnitIds,
    isDraggingFormation,
    formationDragStart,
    formationDragEnd,
    formationConfig,
    addUnit,
    toggleUnitSelection,
    clearSelection,
    moveSelectedUnits,
    startFormationDrag,
    updateFormationDrag,
    endFormationDrag,
    setFormationType,
    setSpreadType,
    toggleIndividualPaths,
    updateUnits,
  } = useGameStore()

  // Animation loop
  useEffect(() => {
    const interval = setInterval(() => {
      updateUnits()
    }, 16) // ~60 FPS
    
    return () => clearInterval(interval)
  }, [updateUnits])

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button !== 0) return // Only left click
    
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return
    
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // Check if shift key is held (for adding units)
    if (e.shiftKey) {
      addUnit({ x, y })
      return
    }
    
    // If no units clicked, clear selection
    if (e.target === svgRef.current) {
      clearSelection()
    }
  }

  // Handle unit click
  const handleUnitClick = (unitId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    
    if (e.ctrlKey || e.metaKey) {
      toggleUnitSelection(unitId)
    } else {
      useGameStore.getState().selectUnits([unitId])
    }
  }

  // Handle right mouse button for movement
  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button === 2) {
      e.preventDefault()
      setIsRightMouseDown(true)
      
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect) return
      
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      
      if (selectedUnitIds.length > 0) {
        startFormationDrag({ x, y })
      }
    }
  }

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isRightMouseDown && isDraggingFormation) {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect) return
      
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      
      updateFormationDrag({ x, y })
    }
  }

  const handleMouseUp = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button === 2 && isRightMouseDown) {
      setIsRightMouseDown(false)
      
      if (isDraggingFormation) {
        endFormationDrag()
      } else {
        // Simple right-click without drag
        const rect = svgRef.current?.getBoundingClientRect()
        if (!rect) return
        
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top
        
        if (selectedUnitIds.length > 0) {
          moveSelectedUnits({ x, y })
        }
      }
    }
  }

  // Prevent context menu
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
  }

  // Calculate group path if multiple units selected
  const selectedUnits = units.filter((u) => selectedUnitIds.includes(u.id))
  const showGroupPath = selectedUnits.length > 1 && !formationConfig.showIndividualPaths
  
  return (
    <div className="app">
      <div className="controls">
        <h1>Orca RTS - Formation Controls</h1>
        
        <div className="control-section">
          <h3>Instructions</h3>
          <ul>
            <li>Shift + Left Click: Add unit</li>
            <li>Left Click: Select unit</li>
            <li>Ctrl/Cmd + Click: Toggle unit selection</li>
            <li>Right Click: Move selected units</li>
            <li>Right Click + Drag: Set facing direction</li>
          </ul>
        </div>
        
        <div className="control-section">
          <h3>Formation Type</h3>
          <div className="button-group">
            <button
              className={formationConfig.type === 'line' ? 'active' : ''}
              onClick={() => setFormationType('line')}
            >
              Line
            </button>
            <button
              className={formationConfig.type === 'box' ? 'active' : ''}
              onClick={() => setFormationType('box')}
            >
              Box
            </button>
            <button
              className={formationConfig.type === 'wedge' ? 'active' : ''}
              onClick={() => setFormationType('wedge')}
            >
              Wedge
            </button>
          </div>
        </div>
        
        <div className="control-section">
          <h3>Spread</h3>
          <div className="button-group">
            <button
              className={formationConfig.spread === 'tight' ? 'active' : ''}
              onClick={() => setSpreadType('tight')}
            >
              Tight
            </button>
            <button
              className={formationConfig.spread === 'normal' ? 'active' : ''}
              onClick={() => setSpreadType('normal')}
            >
              Normal
            </button>
            <button
              className={formationConfig.spread === 'loose' ? 'active' : ''}
              onClick={() => setSpreadType('loose')}
            >
              Loose
            </button>
          </div>
        </div>
        
        <div className="control-section">
          <h3>Path Visualization</h3>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={formationConfig.showIndividualPaths}
              onChange={toggleIndividualPaths}
            />
            Show Individual Paths
          </label>
        </div>
        
        <div className="control-section">
          <h3>Stats</h3>
          <p>Total Units: {units.length}</p>
          <p>Selected: {selectedUnitIds.length}</p>
          <p>Formation: {formationConfig.type}</p>
          <p>Spread: {formationConfig.spread}</p>
          <p>Facing: {Math.round((formationConfig.facing * 180) / Math.PI)}°</p>
        </div>
      </div>
      
      <div className="canvas-container">
        <svg
          ref={svgRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          onClick={handleCanvasClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onContextMenu={handleContextMenu}
          style={{ border: '2px solid #333', background: '#1a1a1a' }}
        >
          {/* Grid background */}
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path
                d="M 50 0 L 0 0 0 50"
                fill="none"
                stroke="#333"
                strokeWidth="1"
              />
            </pattern>
          </defs>
          <rect width={CANVAS_WIDTH} height={CANVAS_HEIGHT} fill="url(#grid)" />
          
          {/* Group path visualization */}
          {showGroupPath && selectedUnits.some((u) => u.targetPosition) && (
            <>
              {selectedUnits.map((unit) => {
                if (!unit.targetPosition) return null
                
                // Calculate center of selected units
                const centerX =
                  selectedUnits.reduce((sum, u) => sum + u.position.x, 0) /
                  selectedUnits.length
                const centerY =
                  selectedUnits.reduce((sum, u) => sum + u.position.y, 0) /
                  selectedUnits.length
                
                const targetCenterX =
                  selectedUnits
                    .filter((u) => u.targetPosition)
                    .reduce((sum, u) => sum + (u.targetPosition?.x || 0), 0) /
                  selectedUnits.filter((u) => u.targetPosition).length
                const targetCenterY =
                  selectedUnits
                    .filter((u) => u.targetPosition)
                    .reduce((sum, u) => sum + (u.targetPosition?.y || 0), 0) /
                  selectedUnits.filter((u) => u.targetPosition).length
                
                return (
                  <line
                    key={`group-path-${unit.id}`}
                    x1={centerX}
                    y1={centerY}
                    x2={targetCenterX}
                    y2={targetCenterY}
                    stroke="#ffffff"
                    strokeWidth="3"
                    strokeDasharray="10,5"
                    opacity="0.4"
                  />
                )
              })[0]}
            </>
          )}
          
          {/* Formation drag indicator */}
          {isDraggingFormation && formationDragStart && formationDragEnd && (
            <>
              <line
                x1={formationDragStart.x}
                y1={formationDragStart.y}
                x2={formationDragEnd.x}
                y2={formationDragEnd.y}
                stroke="#ffff00"
                strokeWidth="3"
                strokeDasharray="5,5"
              />
              <circle
                cx={formationDragStart.x}
                cy={formationDragStart.y}
                r="8"
                fill="#ffff00"
                opacity="0.5"
              />
              <circle
                cx={formationDragEnd.x}
                cy={formationDragEnd.y}
                r="6"
                fill="#ffff00"
              />
            </>
          )}
          
          {/* Units */}
          {units.map((unit) => (
            <RTSUnit
              key={unit.id}
              unit={unit}
              showPath={formationConfig.showIndividualPaths || selectedUnitIds.length === 1}
              onClick={(e) => handleUnitClick(unit.id, e)}
            />
          ))}
        </svg>
      </div>
    </div>
  )
}

export default App
