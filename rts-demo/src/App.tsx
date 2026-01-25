import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Unit {
  id: string
  x: number
  y: number
  type: 'friendly' | 'enemy'
  name: string
  health: number
  maxHealth: number
}

interface SelectionBox {
  startX: number
  startY: number
  endX: number
  endY: number
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [units, setUnits] = useState<Unit[]>([
    // Friendly units (blue)
    { id: 'f1', x: 100, y: 100, type: 'friendly', name: 'Soldier 1', health: 100, maxHealth: 100 },
    { id: 'f2', x: 150, y: 120, type: 'friendly', name: 'Soldier 2', health: 85, maxHealth: 100 },
    { id: 'f3', x: 200, y: 80, type: 'friendly', name: 'Tank 1', health: 200, maxHealth: 200 },
    { id: 'f4', x: 120, y: 180, type: 'friendly', name: 'Soldier 3', health: 100, maxHealth: 100 },
    // Enemy units (red)
    { id: 'e1', x: 500, y: 150, type: 'enemy', name: 'Enemy Soldier 1', health: 90, maxHealth: 100 },
    { id: 'e2', x: 550, y: 200, type: 'enemy', name: 'Enemy Soldier 2', health: 100, maxHealth: 100 },
    { id: 'e3', x: 600, y: 120, type: 'enemy', name: 'Enemy Tank 1', health: 180, maxHealth: 200 },
    { id: 'e4', x: 520, y: 250, type: 'enemy', name: 'Enemy Scout 1', health: 75, maxHealth: 75 },
  ])
  
  const [selectedUnits, setSelectedUnits] = useState<string[]>([])
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null)
  const [hoveredUnit, setHoveredUnit] = useState<string | null>(null)

  const UNIT_SIZE = 30

  // Draw the canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    ctx.strokeStyle = '#e0e0e0'
    ctx.lineWidth = 1
    for (let i = 0; i < canvas.width; i += 50) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, canvas.height)
      ctx.stroke()
    }
    for (let i = 0; i < canvas.height; i += 50) {
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(canvas.width, i)
      ctx.stroke()
    }

    // Draw units
    units.forEach(unit => {
      const isSelected = selectedUnits.includes(unit.id)
      const isHovered = hoveredUnit === unit.id

      // Unit color
      ctx.fillStyle = unit.type === 'friendly' ? '#4CAF50' : '#f44336'
      ctx.beginPath()
      ctx.arc(unit.x, unit.y, UNIT_SIZE / 2, 0, Math.PI * 2)
      ctx.fill()

      // Selection indicator
      if (isSelected) {
        ctx.strokeStyle = unit.type === 'friendly' ? '#2196F3' : '#FF9800'
        ctx.lineWidth = 3
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.arc(unit.x, unit.y, UNIT_SIZE / 2 + 5, 0, Math.PI * 2)
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Hover indicator
      if (isHovered) {
        ctx.strokeStyle = '#FFF'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(unit.x, unit.y, UNIT_SIZE / 2 + 8, 0, Math.PI * 2)
        ctx.stroke()
      }

      // Health bar
      const healthBarWidth = UNIT_SIZE
      const healthBarHeight = 4
      const healthPercentage = unit.health / unit.maxHealth
      
      ctx.fillStyle = '#333'
      ctx.fillRect(unit.x - healthBarWidth / 2, unit.y - UNIT_SIZE / 2 - 10, healthBarWidth, healthBarHeight)
      
      ctx.fillStyle = healthPercentage > 0.5 ? '#4CAF50' : healthPercentage > 0.25 ? '#FF9800' : '#f44336'
      ctx.fillRect(unit.x - healthBarWidth / 2, unit.y - UNIT_SIZE / 2 - 10, healthBarWidth * healthPercentage, healthBarHeight)
    })

    // Draw selection box
    if (selectionBox) {
      const { startX, startY, endX, endY } = selectionBox
      const width = endX - startX
      const height = endY - startY

      // Determine if selecting mostly friendly or enemy units
      const unitsInBox = getUnitsInBox(startX, startY, endX, endY)
      const hasEnemy = unitsInBox.some(u => u.type === 'enemy')
      const hasFriendly = unitsInBox.some(u => u.type === 'friendly')
      
      // Different colors for different selection types
      let fillColor = 'rgba(33, 150, 243, 0.1)' // Default blue
      let strokeColor = '#2196F3'
      
      if (hasEnemy && !hasFriendly) {
        fillColor = 'rgba(255, 152, 0, 0.1)' // Orange for enemy
        strokeColor = '#FF9800'
      } else if (hasEnemy && hasFriendly) {
        fillColor = 'rgba(156, 39, 176, 0.1)' // Purple for mixed
        strokeColor = '#9C27B0'
      }

      ctx.fillStyle = fillColor
      ctx.fillRect(startX, startY, width, height)
      
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = 2
      ctx.setLineDash([5, 3])
      ctx.strokeRect(startX, startY, width, height)
      ctx.setLineDash([])
    }
  }, [units, selectedUnits, selectionBox, hoveredUnit])

  const getUnitsInBox = (x1: number, y1: number, x2: number, y2: number): Unit[] => {
    const left = Math.min(x1, x2)
    const right = Math.max(x1, x2)
    const top = Math.min(y1, y2)
    const bottom = Math.max(y1, y2)

    return units.filter(unit => {
      return unit.x >= left && unit.x <= right && unit.y >= top && unit.y <= bottom
    })
  }

  const getUnitAtPosition = (x: number, y: number): Unit | null => {
    return units.find(unit => {
      const dx = unit.x - x
      const dy = unit.y - y
      return Math.sqrt(dx * dx + dy * dy) <= UNIT_SIZE / 2
    }) || null
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // Check if clicking on a unit
    const clickedUnit = getUnitAtPosition(x, y)
    
    if (clickedUnit) {
      // Single unit selection (toggle)
      if (e.ctrlKey || e.metaKey) {
        setSelectedUnits(prev => 
          prev.includes(clickedUnit.id) 
            ? prev.filter(id => id !== clickedUnit.id)
            : [...prev, clickedUnit.id]
        )
      } else {
        setSelectedUnits([clickedUnit.id])
      }
    } else {
      // Start marquee selection
      setIsSelecting(true)
      setSelectionBox({ startX: x, startY: y, endX: x, endY: y })
      if (!e.shiftKey && !e.ctrlKey) {
        setSelectedUnits([])
      }
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // Update hover state
    const hoveredUnit = getUnitAtPosition(x, y)
    setHoveredUnit(hoveredUnit?.id || null)

    // Update selection box
    if (isSelecting && selectionBox) {
      setSelectionBox({
        ...selectionBox,
        endX: x,
        endY: y
      })
    }
  }

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isSelecting && selectionBox) {
      const { startX, startY, endX, endY } = selectionBox
      const unitsInBox = getUnitsInBox(startX, startY, endX, endY)
      
      if (e.shiftKey || e.ctrlKey) {
        // Add to selection
        setSelectedUnits(prev => {
          const newSelection = [...prev]
          unitsInBox.forEach(unit => {
            if (!newSelection.includes(unit.id)) {
              newSelection.push(unit.id)
            }
          })
          return newSelection
        })
      } else {
        // Replace selection
        setSelectedUnits(unitsInBox.map(u => u.id))
      }
      
      setSelectionBox(null)
    }
    setIsSelecting(false)
  }

  const handleMouseLeave = () => {
    setIsSelecting(false)
    setSelectionBox(null)
    setHoveredUnit(null)
  }

  // Get selected unit objects
  const selectedUnitObjects = units.filter(u => selectedUnits.includes(u.id))
  const hasEnemySelected = selectedUnitObjects.some(u => u.type === 'enemy')
  const hasFriendlySelected = selectedUnitObjects.some(u => u.type === 'friendly')

  return (
    <div className="app">
      <div className="header">
        <h1>Orca RTS Demo - Marquee Selection</h1>
        <div className="legend">
          <div className="legend-item">
            <div className="color-box friendly"></div>
            <span>Friendly Units (Blue Selection)</span>
          </div>
          <div className="legend-item">
            <div className="color-box enemy"></div>
            <span>Enemy Units (Orange Selection)</span>
          </div>
        </div>
      </div>
      
      <div className="main-content">
        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
          />
          <div className="instructions">
            Click and drag to select units | Ctrl+Click for multi-select | Shift+Drag to add to selection
          </div>
        </div>

        <div className="side-panel">
          <div className="info-panel">
            <h2>Selected Units ({selectedUnitObjects.length})</h2>
            {selectedUnitObjects.length === 0 ? (
              <p className="no-selection">No units selected</p>
            ) : (
              <div className="unit-list">
                {selectedUnitObjects.map(unit => (
                  <div key={unit.id} className={`unit-card ${unit.type}`}>
                    <div className="unit-header">
                      <span className="unit-name">{unit.name}</span>
                      <span className={`unit-type ${unit.type}`}>
                        {unit.type === 'friendly' ? '🛡️ Friendly' : '⚔️ Enemy'}
                      </span>
                    </div>
                    <div className="unit-stats">
                      <div className="stat">
                        <span>Health:</span>
                        <span className="stat-value">{unit.health}/{unit.maxHealth}</span>
                      </div>
                      <div className="health-bar-container">
                        <div 
                          className="health-bar-fill"
                          style={{ 
                            width: `${(unit.health / unit.maxHealth) * 100}%`,
                            backgroundColor: unit.health / unit.maxHealth > 0.5 ? '#4CAF50' : 
                                           unit.health / unit.maxHealth > 0.25 ? '#FF9800' : '#f44336'
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="command-panel">
            <h3>Commands</h3>
            {hasEnemySelected && (
              <div className="warning-message">
                ⚠️ Enemy units selected - Commands disabled
              </div>
            )}
            <div className="command-buttons">
              <button 
                disabled={selectedUnitObjects.length === 0 || hasEnemySelected}
                className="command-btn move"
              >
                Move
              </button>
              <button 
                disabled={selectedUnitObjects.length === 0 || hasEnemySelected}
                className="command-btn attack"
              >
                Attack
              </button>
              <button 
                disabled={selectedUnitObjects.length === 0 || hasEnemySelected}
                className="command-btn patrol"
              >
                Patrol
              </button>
              <button 
                disabled={selectedUnitObjects.length === 0 || hasEnemySelected}
                className="command-btn stop"
              >
                Stop
              </button>
            </div>
            {hasFriendlySelected && !hasEnemySelected && (
              <div className="success-message">
                ✓ {selectedUnitObjects.length} friendly unit(s) ready for commands
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
