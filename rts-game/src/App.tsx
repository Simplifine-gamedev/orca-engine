import { useEffect, useRef, useState } from 'react';
import { useGameStore } from './store/gameStore';
import { RTSUnit } from './units/RTSUnit';
import { Position } from './types';
import './App.css';

interface BoxSelection {
  start: Position;
  end: Position;
}

function App() {
  const {
    units,
    selectedUnitIds,
    controlGroups,
    currentActionType,
    selectUnit,
    selectUnits,
    deselectAll,
    saveControlGroup,
    recallControlGroup,
    cycleSelectedUnits,
    moveUnitsTo,
    attackUnit,
    updateUnits,
    initializeUnits,
  } = useGameStore();

  const canvasRef = useRef<HTMLDivElement>(null);
  const [boxSelection, setBoxSelection] = useState<BoxSelection | null>(null);
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [cursorPosition, setCursorPosition] = useState<Position>({ x: 0, y: 0 });
  const gameLoopRef = useRef<number>();

  // Initialize units on mount
  useEffect(() => {
    initializeUnits();
  }, [initializeUnits]);

  // Game loop for unit movement
  useEffect(() => {
    const gameLoop = () => {
      updateUnits();
      gameLoopRef.current = requestAnimationFrame(gameLoop);
    };
    
    gameLoopRef.current = requestAnimationFrame(gameLoop);
    
    return () => {
      if (gameLoopRef.current) {
        cancelAnimationFrame(gameLoopRef.current);
      }
    };
  }, [updateUnits]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Control groups (Ctrl+1-9 to save, 1-9 to recall)
      if (e.key >= '1' && e.key <= '9') {
        const groupNumber = parseInt(e.key);
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          saveControlGroup(groupNumber);
        } else {
          recallControlGroup(groupNumber);
        }
      }
      
      // Tab to cycle through selected units
      if (e.key === 'Tab') {
        e.preventDefault();
        cycleSelectedUnits();
      }

      // Escape to deselect all
      if (e.key === 'Escape') {
        deselectAll();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [saveControlGroup, recallControlGroup, cycleSelectedUnits, deselectAll]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const target = e.target as HTMLElement;
    const unitId = target.closest('[data-unit-id]')?.getAttribute('data-unit-id');

    if (unitId) {
      // Clicked on a unit
      const unit = units.find(u => u.id === unitId);
      if (unit) {
        if (unit.team === 'player') {
          selectUnit(unitId, e.shiftKey);
        }
      }
    } else {
      // Start box selection
      setIsMouseDown(true);
      setBoxSelection({
        start: { x, y },
        end: { x, y },
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCursorPosition({ x, y });

    if (isMouseDown && boxSelection) {
      setBoxSelection({
        ...boxSelection,
        end: { x, y },
      });
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const target = e.target as HTMLElement;
    const unitId = target.closest('[data-unit-id]')?.getAttribute('data-unit-id');

    if (isMouseDown && boxSelection) {
      // Complete box selection
      const minX = Math.min(boxSelection.start.x, boxSelection.end.x);
      const maxX = Math.max(boxSelection.start.x, boxSelection.end.x);
      const minY = Math.min(boxSelection.start.y, boxSelection.end.y);
      const maxY = Math.max(boxSelection.start.y, boxSelection.end.y);

      const selectedIds = units
        .filter(u => 
          u.team === 'player' &&
          u.position.x >= minX &&
          u.position.x <= maxX &&
          u.position.y >= minY &&
          u.position.y <= maxY
        )
        .map(u => u.id);

      if (selectedIds.length > 0) {
        selectUnits(selectedIds);
      }

      setBoxSelection(null);
      setIsMouseDown(false);
    }
  };

  const handleRightClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (!canvasRef.current || selectedUnitIds.length === 0) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const target = e.target as HTMLElement;
    const targetUnitId = target.closest('[data-unit-id]')?.getAttribute('data-unit-id');

    if (targetUnitId) {
      const targetUnit = units.find(u => u.id === targetUnitId);
      if (targetUnit && targetUnit.team === 'enemy') {
        // Attack enemy unit
        attackUnit(targetUnitId);
        return;
      }
    }

    // Move to position
    moveUnitsTo({ x, y });
  };

  const getCursorStyle = (): string => {
    const target = document.elementFromPoint(cursorPosition.x, cursorPosition.y);
    const unitId = target?.closest('[data-unit-id]')?.getAttribute('data-unit-id');
    
    if (unitId) {
      const unit = units.find(u => u.id === unitId);
      if (unit?.team === 'enemy' && selectedUnitIds.length > 0) {
        return 'crosshair';
      }
      if (unit?.team === 'player') {
        return 'pointer';
      }
    }
    
    return 'default';
  };

  return (
    <div className="app">
      <div className="header">
        <h1>Orca RTS</h1>
        <div className="stats">
          <span>Units Selected: {selectedUnitIds.length}</span>
          <span>Total Units: {units.filter(u => u.health > 0).length}</span>
        </div>
      </div>

      <div className="controls-info">
        <div className="control-group">
          <strong>Selection:</strong>
          <span>Left Click - Select | Shift+Click - Add/Remove | Drag - Box Select</span>
        </div>
        <div className="control-group">
          <strong>Commands:</strong>
          <span>Right Click - Move/Attack | Tab - Cycle Units | ESC - Deselect</span>
        </div>
        <div className="control-group">
          <strong>Control Groups:</strong>
          <span>Ctrl+1-9 - Save Group | 1-9 - Recall Group</span>
        </div>
      </div>

      <div className="control-groups-display">
        {Object.entries(controlGroups).map(([key, unitIds]) => (
          <div key={key} className="control-group-item">
            <span className="group-number">{key}</span>
            <span className="group-count">{unitIds.length} units</span>
          </div>
        ))}
      </div>

      {currentActionType && (
        <div className={`action-indicator ${currentActionType}`}>
          {currentActionType === 'move' ? '📍 Moving' : '⚔️ Attacking'}
        </div>
      )}

      <div
        ref={canvasRef}
        className="game-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={handleRightClick}
        style={{ cursor: getCursorStyle() }}
      >
        {/* Render units */}
        {units.map(unit => (
          <RTSUnit key={unit.id} unit={unit} />
        ))}

        {/* Box selection visualization */}
        {boxSelection && (
          <div
            className="box-selection"
            style={{
              left: Math.min(boxSelection.start.x, boxSelection.end.x),
              top: Math.min(boxSelection.start.y, boxSelection.end.y),
              width: Math.abs(boxSelection.end.x - boxSelection.start.x),
              height: Math.abs(boxSelection.end.y - boxSelection.start.y),
            }}
          />
        )}
      </div>

      <div className="footer">
        <p>Click and drag to select multiple units. Right-click to move or attack.</p>
      </div>
    </div>
  );
}

export default App;
