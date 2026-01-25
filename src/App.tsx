// Main App with input handling and UI

import React, { useEffect, useState, useRef } from 'react';
import { gameStore } from './store/gameStore';
import { GameState, Vector2, FormationType, SpreadType } from './types';
import { RTSUnit, FormationPreview, GroupPath } from './units/RTSUnit';
import './App.css';

const App: React.FC = () => {
  const [gameState, setGameState] = useState<GameState>(gameStore.getState());
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionStart, setSelectionStart] = useState<Vector2 | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<Vector2 | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const animationRef = useRef<number>();
  const lastTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    const unsubscribe = gameStore.subscribe(() => {
      setGameState(gameStore.getState());
    });

    // Animation loop
    const animate = () => {
      const now = Date.now();
      const deltaTime = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;
      
      gameStore.updateUnits(deltaTime);
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      unsubscribe();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const getSvgCoordinates = (event: React.MouseEvent<SVGSVGElement>): Vector2 => {
    if (!svgRef.current) return { x: 0, y: 0 };
    
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  };

  const handleMouseDown = (event: React.MouseEvent<SVGSVGElement>) => {
    const pos = getSvgCoordinates(event);
    const shiftKey = event.shiftKey;
    
    // Check if clicking on a unit
    const clickedUnit = gameState.units.find(unit => {
      const dx = unit.position.x - pos.x;
      const dy = unit.position.y - pos.y;
      return Math.sqrt(dx * dx + dy * dy) < 20;
    });

    if (clickedUnit) {
      gameStore.selectUnit(clickedUnit.id, shiftKey);
    } else if (event.button === 0) {
      // Left click - start selection box
      if (!shiftKey) {
        gameStore.deselectAll();
      }
      setIsSelecting(true);
      setSelectionStart(pos);
      setSelectionEnd(pos);
    }
  };

  const handleMouseMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const pos = getSvgCoordinates(event);
    
    if (isSelecting && selectionStart) {
      setSelectionEnd(pos);
    } else if (gameState.isDraggingFormation) {
      gameStore.updateFormationDrag(pos);
    }
  };

  const handleMouseUp = (event: React.MouseEvent<SVGSVGElement>) => {
    const pos = getSvgCoordinates(event);
    
    if (isSelecting && selectionStart && selectionEnd) {
      gameStore.selectUnitsInArea(selectionStart, selectionEnd);
      setIsSelecting(false);
      setSelectionStart(null);
      setSelectionEnd(null);
    } else if (gameState.isDraggingFormation) {
      gameStore.endFormationDrag();
    }
  };

  const handleRightClick = (event: React.MouseEvent<SVGSVGElement>) => {
    event.preventDefault();
    const pos = getSvgCoordinates(event);
    
    if (gameState.selectedUnits.length > 0) {
      if (event.shiftKey) {
        // Shift + Right click: Start formation drag
        gameStore.startFormationDrag(pos);
      } else {
        // Regular right click: Move to position
        gameStore.moveSelectedUnits(pos);
      }
    }
  };

  const groupPath = gameStore.getGroupPath();

  return (
    <div className="app">
      <div className="header">
        <h1>RTS Formation Control Demo</h1>
        <div className="unit-count">
          Units Selected: {gameState.selectedUnits.length} / {gameState.units.length}
        </div>
      </div>

      <div className="main-container">
        <div className="controls-panel">
          <div className="control-section">
            <h3>Formation Type</h3>
            <div className="button-group">
              <button
                className={gameState.formationSettings.type === 'none' ? 'active' : ''}
                onClick={() => gameStore.setFormationType('none')}
              >
                None
              </button>
              <button
                className={gameState.formationSettings.type === 'line' ? 'active' : ''}
                onClick={() => gameStore.setFormationType('line')}
              >
                Line
              </button>
              <button
                className={gameState.formationSettings.type === 'box' ? 'active' : ''}
                onClick={() => gameStore.setFormationType('box')}
              >
                Box
              </button>
              <button
                className={gameState.formationSettings.type === 'wedge' ? 'active' : ''}
                onClick={() => gameStore.setFormationType('wedge')}
              >
                Wedge
              </button>
            </div>
          </div>

          <div className="control-section">
            <h3>Spread</h3>
            <div className="button-group">
              <button
                className={gameState.formationSettings.spread === 'tight' ? 'active' : ''}
                onClick={() => gameStore.setSpread('tight')}
              >
                Tight
              </button>
              <button
                className={gameState.formationSettings.spread === 'normal' ? 'active' : ''}
                onClick={() => gameStore.setSpread('normal')}
              >
                Normal
              </button>
              <button
                className={gameState.formationSettings.spread === 'loose' ? 'active' : ''}
                onClick={() => gameStore.setSpread('loose')}
              >
                Loose
              </button>
            </div>
          </div>

          <div className="control-section">
            <h3>Path Visualization</h3>
            <div className="checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={gameState.formationSettings.showIndividualPaths}
                  onChange={() => gameStore.toggleIndividualPaths()}
                />
                Show Individual Paths
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={gameState.formationSettings.showGroupPath}
                  onChange={() => gameStore.toggleGroupPath()}
                />
                Show Group Path
              </label>
            </div>
          </div>

          <div className="control-section instructions">
            <h3>Controls</h3>
            <ul>
              <li><strong>Left Click:</strong> Select unit</li>
              <li><strong>Drag:</strong> Box select multiple units</li>
              <li><strong>Shift + Click:</strong> Add to selection</li>
              <li><strong>Right Click:</strong> Move selected units</li>
              <li><strong>Shift + Right Click + Drag:</strong> Set facing direction</li>
            </ul>
          </div>
        </div>

        <div className="game-viewport">
          <svg
            ref={svgRef}
            width="100%"
            height="100%"
            viewBox="0 0 800 600"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onContextMenu={handleRightClick}
            style={{ background: '#1e293b' }}
          >
            {/* Grid background */}
            <defs>
              <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#334155" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="800" height="600" fill="url(#grid)" />

            {/* Group path (drawn first, behind units) */}
            {groupPath && <GroupPath path={groupPath} />}

            {/* Units */}
            {gameState.units.map(unit => (
              <RTSUnit
                key={unit.id}
                unit={unit}
                showPath={gameState.formationSettings.showIndividualPaths}
              />
            ))}

            {/* Formation preview during drag */}
            {gameState.isDraggingFormation &&
             gameState.formationDragStart &&
             gameState.formationDragEnd && (
              <FormationPreview
                dragStart={gameState.formationDragStart}
                dragEnd={gameState.formationDragEnd}
              />
            )}

            {/* Selection box */}
            {isSelecting && selectionStart && selectionEnd && (
              <rect
                x={Math.min(selectionStart.x, selectionEnd.x)}
                y={Math.min(selectionStart.y, selectionEnd.y)}
                width={Math.abs(selectionEnd.x - selectionStart.x)}
                height={Math.abs(selectionEnd.y - selectionStart.y)}
                fill="#3b82f6"
                fillOpacity="0.2"
                stroke="#3b82f6"
                strokeWidth="2"
              />
            )}
          </svg>
        </div>
      </div>
    </div>
  );
};

export default App;
