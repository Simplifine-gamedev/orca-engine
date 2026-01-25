import React, { useEffect, useCallback } from 'react';
import { useGameStore, BUILDING_TYPES } from './store/gameStore';
import { Building } from './buildings/Building';

const GRID_SIZE = 32; // pixels per grid cell
const MAP_WIDTH = 20; // grid cells
const MAP_HEIGHT = 15; // grid cells

function App() {
  const {
    buildings,
    selectedBuildingType,
    ghostPosition,
    ghostRotation,
    selectBuildingType,
    setGhostPosition,
    rotateGhost,
    placeBuilding,
    removeBuilding,
  } = useGameStore();
  
  // Handle keyboard input for rotation (R key)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'r' || e.key === 'R') {
        if (selectedBuildingType && ghostPosition) {
          rotateGhost();
        }
      } else if (e.key === 'Escape') {
        selectBuildingType(null);
        setGhostPosition(null);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedBuildingType, ghostPosition, rotateGhost, selectBuildingType, setGhostPosition]);
  
  // Handle mouse wheel for rotation
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (selectedBuildingType && ghostPosition) {
      e.preventDefault();
      rotateGhost();
    }
  }, [selectedBuildingType, ghostPosition, rotateGhost]);
  
  // Handle mouse movement for ghost preview
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!selectedBuildingType) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / GRID_SIZE);
    const y = Math.floor((e.clientY - rect.top) / GRID_SIZE);
    
    setGhostPosition({ x, y });
  }, [selectedBuildingType, setGhostPosition]);
  
  // Handle click to place building
  const handleMapClick = useCallback(() => {
    if (selectedBuildingType && ghostPosition) {
      placeBuilding();
    }
  }, [selectedBuildingType, ghostPosition, placeBuilding]);
  
  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      backgroundColor: '#1a1a1a',
      color: '#fff',
      fontFamily: 'Arial, sans-serif',
    }}>
      {/* Header */}
      <div style={{ 
        padding: '20px', 
        backgroundColor: '#2a2a2a',
        borderBottom: '2px solid #444',
      }}>
        <h1 style={{ margin: '0 0 10px 0' }}>Orca RTS - Building Placement</h1>
        <p style={{ margin: '5px 0', fontSize: '14px', color: '#aaa' }}>
          Select a building, move your mouse to position it, press <strong>R</strong> or <strong>scroll wheel</strong> to rotate, and click to place.
        </p>
      </div>
      
      {/* Building Selection Toolbar */}
      <div style={{ 
        padding: '15px 20px', 
        backgroundColor: '#333',
        borderBottom: '2px solid #444',
        display: 'flex',
        gap: '10px',
        flexWrap: 'wrap',
      }}>
        {BUILDING_TYPES.map((buildingType) => (
          <button
            key={buildingType.id}
            onClick={() => selectBuildingType(buildingType)}
            style={{
              padding: '10px 20px',
              backgroundColor: selectedBuildingType?.id === buildingType.id ? buildingType.color : '#555',
              color: '#fff',
              border: selectedBuildingType?.id === buildingType.id ? '3px solid #fff' : '2px solid #777',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
              transition: 'all 0.2s ease',
            }}
          >
            {buildingType.name} ({buildingType.width}×{buildingType.height})
          </button>
        ))}
        {selectedBuildingType && (
          <button
            onClick={() => selectBuildingType(null)}
            style={{
              padding: '10px 20px',
              backgroundColor: '#d9534f',
              color: '#fff',
              border: '2px solid #c9302c',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            Cancel (ESC)
          </button>
        )}
      </div>
      
      {/* Status Bar */}
      {selectedBuildingType && (
        <div style={{
          padding: '10px 20px',
          backgroundColor: '#2a4a2a',
          borderBottom: '2px solid #444',
          fontSize: '14px',
        }}>
          Placing: <strong>{selectedBuildingType.name}</strong> | 
          Rotation: <strong>{ghostRotation}°</strong> | 
          Press <strong>R</strong> or use <strong>scroll wheel</strong> to rotate
        </div>
      )}
      
      {/* Game Map */}
      <div style={{ 
        flex: 1, 
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '20px',
        overflow: 'auto',
      }}>
        <div
          style={{
            position: 'relative',
            width: MAP_WIDTH * GRID_SIZE,
            height: MAP_HEIGHT * GRID_SIZE,
            backgroundColor: '#2d5016',
            backgroundImage: `
              linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
            `,
            backgroundSize: `${GRID_SIZE}px ${GRID_SIZE}px`,
            border: '3px solid #444',
            cursor: selectedBuildingType ? 'crosshair' : 'default',
          }}
          onMouseMove={handleMouseMove}
          onClick={handleMapClick}
          onWheel={handleWheel}
        >
          {/* Render placed buildings */}
          {buildings.map((building) => (
            <Building
              key={building.id}
              building={building}
              x={building.x}
              y={building.y}
              rotation={building.rotation}
              onClick={() => {
                if (!selectedBuildingType) {
                  if (window.confirm(`Remove ${building.type.name}?`)) {
                    removeBuilding(building.id);
                  }
                }
              }}
            />
          ))}
          
          {/* Render ghost preview */}
          {selectedBuildingType && ghostPosition && (
            <Building
              buildingType={selectedBuildingType}
              x={ghostPosition.x}
              y={ghostPosition.y}
              rotation={ghostRotation}
              isGhost={true}
            />
          )}
        </div>
      </div>
      
      {/* Footer with instructions */}
      <div style={{
        padding: '15px 20px',
        backgroundColor: '#2a2a2a',
        borderTop: '2px solid #444',
        fontSize: '12px',
        color: '#aaa',
      }}>
        <strong>Controls:</strong> Select building → Move mouse → Press R or scroll to rotate → Click to place | 
        Click on placed buildings to remove them | Press ESC to cancel
      </div>
    </div>
  );
}

export default App;
