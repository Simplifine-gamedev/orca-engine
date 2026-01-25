import React, { useEffect, useRef, useState } from 'react';
import { useGameStore } from './store/gameStore';
import RTSUnit from './units/RTSUnit';
import { Unit, Position } from './types/unit';

const App: React.FC = () => {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [isCtrlPressed, setIsCtrlPressed] = useState(false);
  const [isShiftPressed, setIsShiftPressed] = useState(false);
  const [cursorMode, setCursorMode] = useState<'select' | 'move' | 'attack'>('select');
  
  const {
    units,
    selectedUnitIds,
    selectionBox,
    isSelecting,
    controlGroups,
    showMoveIndicator,
    moveIndicatorPosition,
    addUnit,
    selectUnits,
    addToSelection,
    clearSelection,
    startSelection,
    updateSelection,
    endSelection,
    assignControlGroup,
    selectControlGroup,
    cycleSelectedUnits,
    moveUnits,
    updateUnitPositions,
    showMoveIndicatorAt,
    hideMoveIndicator,
    getSelectedUnits
  } = useGameStore();
  
  // Initialize some demo units
  useEffect(() => {
    const initialUnits: Unit[] = [
      {
        id: 'unit-1',
        position: { x: 200, y: 200 },
        health: 100,
        maxHealth: 100,
        type: 'warrior',
        team: 'player',
        isMoving: false,
        targetPosition: null,
        speed: 2
      },
      {
        id: 'unit-2',
        position: { x: 300, y: 200 },
        health: 80,
        maxHealth: 100,
        type: 'archer',
        team: 'player',
        isMoving: false,
        targetPosition: null,
        speed: 2.5
      },
      {
        id: 'unit-3',
        position: { x: 250, y: 300 },
        health: 60,
        maxHealth: 100,
        type: 'mage',
        team: 'player',
        isMoving: false,
        targetPosition: null,
        speed: 1.8
      },
      {
        id: 'unit-4',
        position: { x: 400, y: 250 },
        health: 100,
        maxHealth: 100,
        type: 'warrior',
        team: 'player',
        isMoving: false,
        targetPosition: null,
        speed: 2
      },
      {
        id: 'unit-5',
        position: { x: 500, y: 300 },
        health: 90,
        maxHealth: 100,
        type: 'archer',
        team: 'player',
        isMoving: false,
        targetPosition: null,
        speed: 2.5
      }
    ];
    
    initialUnits.forEach(addUnit);
  }, []);
  
  // Game loop for unit movement
  useEffect(() => {
    const interval = setInterval(() => {
      updateUnitPositions();
    }, 16); // ~60 FPS
    
    return () => clearInterval(interval);
  }, []);
  
  // Keyboard event handlers
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Track modifier keys
      if (e.key === 'Control') setIsCtrlPressed(true);
      if (e.key === 'Shift') setIsShiftPressed(true);
      
      // Tab to cycle through selected units
      if (e.key === 'Tab') {
        e.preventDefault();
        cycleSelectedUnits();
      }
      
      // Escape to clear selection
      if (e.key === 'Escape') {
        clearSelection();
      }
      
      // Control groups (Ctrl + 1-9 to assign, 1-9 to select)
      const numKey = parseInt(e.key);
      if (!isNaN(numKey) && numKey >= 1 && numKey <= 9) {
        if (e.ctrlKey) {
          e.preventDefault();
          // Assign selected units to control group
          assignControlGroup(numKey, selectedUnitIds);
          showNotification(`Control group ${numKey} assigned`);
        } else {
          // Select control group
          selectControlGroup(numKey, e.shiftKey);
        }
      }
    };
    
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Control') setIsCtrlPressed(false);
      if (e.key === 'Shift') setIsShiftPressed(false);
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [selectedUnitIds, isCtrlPressed, isShiftPressed]);
  
  // Mouse event handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if clicking on a unit
    const clickedUnit = units.find(unit => {
      const dx = unit.position.x - x;
      const dy = unit.position.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 25; // Unit radius
    });
    
    if (clickedUnit && clickedUnit.team === 'player') {
      // Unit clicked
      if (isShiftPressed) {
        // Add to selection
        if (selectedUnitIds.includes(clickedUnit.id)) {
          // Deselect if already selected
          selectUnits(selectedUnitIds.filter(id => id !== clickedUnit.id));
        } else {
          addToSelection([clickedUnit.id]);
        }
      } else {
        // Replace selection
        selectUnits([clickedUnit.id]);
      }
    } else {
      // Start box selection
      if (!isShiftPressed) {
        clearSelection();
      }
      startSelection(x, y);
    }
  };
  
  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (isSelecting) {
      updateSelection(x, y);
    }
    
    // Update cursor mode based on hover
    const hoveredUnit = units.find(unit => {
      const dx = unit.position.x - x;
      const dy = unit.position.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 25;
    });
    
    if (hoveredUnit) {
      if (hoveredUnit.team === 'player') {
        setCursorMode('select');
      } else {
        setCursorMode('attack');
      }
    } else if (selectedUnitIds.length > 0) {
      setCursorMode('move');
    } else {
      setCursorMode('select');
    }
  };
  
  const handleMouseUp = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    
    if (isSelecting) {
      endSelection();
      return;
    }
  };
  
  const handleRightClick = (e: React.MouseEvent) => {
    e.preventDefault();
    
    if (selectedUnitIds.length === 0) return;
    
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if right-clicking on an enemy unit
    const targetUnit = units.find(unit => {
      const dx = unit.position.x - x;
      const dy = unit.position.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 25 && unit.team === 'enemy';
    });
    
    if (targetUnit) {
      // Attack command (not implemented in this version)
      showNotification('Attack command!');
    } else {
      // Move command
      const targetPosition: Position = { x, y };
      
      // Calculate formation positions for multiple units
      const selectedUnits = getSelectedUnits();
      const positions = calculateFormationPositions(targetPosition, selectedUnits.length);
      
      selectedUnits.forEach((unit, index) => {
        moveUnits([unit.id], positions[index]);
      });
      
      // Show move indicator
      showMoveIndicatorAt(targetPosition);
      setTimeout(() => hideMoveIndicator(), 1000);
    }
  };
  
  // Calculate formation positions for multiple units
  const calculateFormationPositions = (center: Position, count: number): Position[] => {
    if (count === 1) return [center];
    
    const positions: Position[] = [];
    const spacing = 60;
    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    
    for (let i = 0; i < count; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      positions.push({
        x: center.x + (col - cols / 2) * spacing,
        y: center.y + (row - rows / 2) * spacing
      });
    }
    
    return positions;
  };
  
  // Notification system
  const [notification, setNotification] = useState<string | null>(null);
  
  const showNotification = (message: string) => {
    setNotification(message);
    setTimeout(() => setNotification(null), 2000);
  };
  
  // Handle unit click
  const handleUnitClick = (unit: Unit, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (isShiftPressed) {
      if (selectedUnitIds.includes(unit.id)) {
        selectUnits(selectedUnitIds.filter(id => id !== unit.id));
      } else {
        addToSelection([unit.id]);
      }
    } else {
      selectUnits([unit.id]);
    }
  };
  
  // Render selection box
  const renderSelectionBox = () => {
    if (!selectionBox) return null;
    
    const { startX, startY, endX, endY } = selectionBox;
    const left = Math.min(startX, endX);
    const top = Math.min(startY, endY);
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);
    
    const style: React.CSSProperties = {
      position: 'absolute',
      left,
      top,
      width,
      height,
      border: '2px dashed #00BFFF',
      backgroundColor: 'rgba(0, 191, 255, 0.1)',
      pointerEvents: 'none',
      zIndex: 10
    };
    
    return <div style={style} />;
  };
  
  // Render move indicator
  const renderMoveIndicator = () => {
    if (!showMoveIndicator || !moveIndicatorPosition) return null;
    
    const style: React.CSSProperties = {
      position: 'absolute',
      left: moveIndicatorPosition.x - 20,
      top: moveIndicatorPosition.y - 20,
      width: 40,
      height: 40,
      border: '3px solid #4ADE80',
      borderRadius: '50%',
      backgroundColor: 'rgba(74, 222, 128, 0.2)',
      pointerEvents: 'none',
      animation: 'expandFade 1s ease-out',
      zIndex: 5
    };
    
    return <div style={style} />;
  };
  
  // Get cursor style
  const getCursorStyle = (): string => {
    switch (cursorMode) {
      case 'move':
        return 'pointer';
      case 'attack':
        return 'crosshair';
      default:
        return 'default';
    }
  };
  
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      backgroundColor: '#1a1a2e',
      overflow: 'hidden',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Game canvas */}
      <div
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={handleRightClick}
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          cursor: getCursorStyle()
        }}
      >
        {/* Background grid */}
        <div style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
          backgroundSize: '50px 50px',
          pointerEvents: 'none'
        }} />
        
        {/* Render units */}
        {units.map(unit => (
          <RTSUnit
            key={unit.id}
            unit={unit}
            isSelected={selectedUnitIds.includes(unit.id)}
            onClick={(e) => handleUnitClick(unit, e)}
          />
        ))}
        
        {/* Selection box */}
        {renderSelectionBox()}
        
        {/* Move indicator */}
        {renderMoveIndicator()}
      </div>
      
      {/* UI Overlay */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        color: 'white',
        backgroundColor: 'rgba(0,0,0,0.7)',
        padding: 20,
        borderRadius: 10,
        backdropFilter: 'blur(10px)',
        maxWidth: 350
      }}>
        <h2 style={{ margin: '0 0 15px 0', fontSize: 24, color: '#FFD700' }}>RTS Unit Control</h2>
        
        <div style={{ marginBottom: 15 }}>
          <strong style={{ color: '#4ADE80' }}>Selected Units:</strong> {selectedUnitIds.length}
        </div>
        
        <div style={{ marginBottom: 15, fontSize: 14, lineHeight: 1.6 }}>
          <div style={{ marginBottom: 8, color: '#00BFFF', fontWeight: 'bold' }}>Controls:</div>
          <div>• <strong>Left Click:</strong> Select unit</div>
          <div>• <strong>Drag:</strong> Box select</div>
          <div>• <strong>Right Click:</strong> Move units</div>
          <div>• <strong>Shift + Click:</strong> Add/remove from selection</div>
          <div>• <strong>Ctrl + 1-9:</strong> Assign control group</div>
          <div>• <strong>1-9:</strong> Select control group</div>
          <div>• <strong>Tab:</strong> Cycle selected units</div>
          <div>• <strong>Esc:</strong> Clear selection</div>
        </div>
        
        <div style={{ fontSize: 12, borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: 10 }}>
          {Object.keys(controlGroups).length > 0 && (
            <>
              <strong style={{ color: '#FFD700' }}>Control Groups:</strong>
              <div style={{ marginTop: 5 }}>
                {Object.entries(controlGroups).map(([key, unitIds]) => (
                  <div key={key}>
                    Group {key}: {unitIds.length} units
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
      
      {/* Notification */}
      {notification && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0,0,0,0.9)',
          color: '#FFD700',
          padding: '20px 40px',
          borderRadius: 10,
          fontSize: 24,
          fontWeight: 'bold',
          border: '2px solid #FFD700',
          boxShadow: '0 0 30px rgba(255,215,0,0.5)',
          zIndex: 1000,
          animation: 'fadeIn 0.3s ease-out'
        }}>
          {notification}
        </div>
      )}
      
      {/* CSS Animations */}
      <style>{`
        @keyframes expandFade {
          0% { transform: scale(0.5); opacity: 1; }
          100% { transform: scale(2); opacity: 0; }
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translate(-50%, -60%); }
          to { opacity: 1; transform: translate(-50%, -50%); }
        }
      `}</style>
    </div>
  );
};

export default App;
