import React, { useEffect, useRef } from 'react'
import { useGameStore, BuildingType, getBuildingSize } from './store/gameStore'
import { Building } from './buildings/Building'
import './App.css'

function App() {
  const canvasRef = useRef<HTMLDivElement>(null)
  const {
    buildings,
    selectedBuildingType,
    ghostBuilding,
    isPlacingBuilding,
    setSelectedBuildingType,
    updateGhostBuilding,
    rotateGhostBuilding,
    placeBuilding,
    cancelPlacement
  } = useGameStore()
  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // R key to rotate building ghost
      if (e.key === 'r' || e.key === 'R') {
        if (isPlacingBuilding) {
          e.preventDefault()
          rotateGhostBuilding()
        }
      }
      
      // ESC key to cancel placement
      if (e.key === 'Escape') {
        if (isPlacingBuilding) {
          e.preventDefault()
          cancelPlacement()
        }
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isPlacingBuilding, rotateGhostBuilding, cancelPlacement])
  
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPlacingBuilding || !canvasRef.current) return
    
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    updateGhostBuilding(x, y)
  }
  
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPlacingBuilding) return
    
    e.preventDefault()
    placeBuilding()
  }
  
  const handleBuildingSelect = (type: BuildingType) => {
    setSelectedBuildingType(type)
  }
  
  return (
    <div className="app">
      <header className="header">
        <h1>Orca RTS</h1>
        <div className="info">
          <span>Press R to rotate buildings</span>
          <span>ESC to cancel</span>
        </div>
      </header>
      
      <div className="game-container">
        <aside className="sidebar">
          <h2>Buildings</h2>
          <div className="building-buttons">
            <button
              className={`building-btn barracks ${selectedBuildingType === 'barracks' ? 'active' : ''}`}
              onClick={() => handleBuildingSelect('barracks')}
            >
              Barracks
            </button>
            <button
              className={`building-btn factory ${selectedBuildingType === 'factory' ? 'active' : ''}`}
              onClick={() => handleBuildingSelect('factory')}
            >
              Factory
            </button>
            <button
              className={`building-btn power-plant ${selectedBuildingType === 'powerPlant' ? 'active' : ''}`}
              onClick={() => handleBuildingSelect('powerPlant')}
            >
              Power Plant
            </button>
            <button
              className={`building-btn mine ${selectedBuildingType === 'mine' ? 'active' : ''}`}
              onClick={() => handleBuildingSelect('mine')}
            >
              Mine
            </button>
          </div>
          
          <div className="stats">
            <h3>Stats</h3>
            <p>Buildings: {buildings.length}</p>
          </div>
        </aside>
        
        <main
          ref={canvasRef}
          className={`game-canvas ${isPlacingBuilding ? 'placing' : ''}`}
          onMouseMove={handleMouseMove}
          onClick={handleClick}
        >
          {buildings.map((building) => (
            <Building key={building.id} building={building} />
          ))}
          
          {ghostBuilding && ghostBuilding.type && (
            <Building
              building={{
                id: 'ghost',
                type: ghostBuilding.type,
                x: ghostBuilding.x,
                y: ghostBuilding.y,
                rotation: ghostBuilding.rotation,
                ...getBuildingSize(ghostBuilding.type)
              }}
              isGhost={true}
            />
          )}
          
          {!isPlacingBuilding && buildings.length === 0 && (
            <div className="empty-state">
              <h2>Welcome to Orca RTS!</h2>
              <p>Select a building from the sidebar to start placing</p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
