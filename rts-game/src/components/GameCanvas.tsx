import React, { useEffect, useRef } from 'react';
import useGameStore from '../store/gameStore';
import Unit from './Unit';
import DamageNumber from '../effects/DamageNumber';
import { Position } from '../types';

const GameCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLDivElement>(null);
  const lastUpdateRef = useRef<number>(Date.now());

  const {
    units,
    damageEvents,
    settings,
    selectUnit,
    moveUnit,
    setTarget,
    updateUnits,
    deselectAll,
  } = useGameStore();

  // Game loop
  useEffect(() => {
    const gameLoop = () => {
      const now = Date.now();
      const deltaTime = now - lastUpdateRef.current;
      lastUpdateRef.current = now;

      updateUnits(deltaTime);
    };

    const intervalId = setInterval(gameLoop, 16); // ~60 FPS

    return () => clearInterval(intervalId);
  }, [updateUnits]);

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (e.button === 2) return; // Ignore right clicks on canvas

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const clickPos: Position = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };

    // Check if clicking on empty space to deselect
    const clickedOnUnit = units.some((unit) => {
      const dx = unit.position.x - clickPos.x;
      const dy = unit.position.y - clickPos.y;
      return Math.sqrt(dx * dx + dy * dy) < 20;
    });

    if (!clickedOnUnit) {
      deselectAll();
    }
  };

  const handleCanvasRightClick = (e: React.MouseEvent) => {
    e.preventDefault();

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const clickPos: Position = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };

    const selectedUnit = units.find((u) => u.isSelected);
    if (!selectedUnit) return;

    // Check if right-clicking on an enemy unit to attack
    const clickedUnit = units.find((unit) => {
      if (unit.team === selectedUnit.team) return false;
      const dx = unit.position.x - clickPos.x;
      const dy = unit.position.y - clickPos.y;
      return Math.sqrt(dx * dx + dy * dy) < 20;
    });

    if (clickedUnit) {
      setTarget(selectedUnit.id, clickedUnit.id);
    } else {
      // Move to position
      moveUnit(selectedUnit.id, clickPos);
    }
  };

  const handleUnitSelect = (id: string) => {
    selectUnit(id);
  };

  const handleUnitRightClick = (id: string) => {
    const selectedUnit = units.find((u) => u.isSelected);
    const targetUnit = units.find((u) => u.id === id);

    if (selectedUnit && targetUnit && selectedUnit.team !== targetUnit.team) {
      setTarget(selectedUnit.id, id);
    }
  };

  const canvasStyle: React.CSSProperties = {
    position: 'relative',
    width: '100%',
    height: '100%',
    backgroundColor: '#2a2a2a',
    backgroundImage: `
      repeating-linear-gradient(
        0deg,
        rgba(255, 255, 255, 0.05) 0px,
        rgba(255, 255, 255, 0.05) 1px,
        transparent 1px,
        transparent 50px
      ),
      repeating-linear-gradient(
        90deg,
        rgba(255, 255, 255, 0.05) 0px,
        rgba(255, 255, 255, 0.05) 1px,
        transparent 1px,
        transparent 50px
      )
    `,
    cursor: 'default',
    overflow: 'hidden',
  };

  return (
    <div
      ref={canvasRef}
      style={canvasStyle}
      onClick={handleCanvasClick}
      onContextMenu={handleCanvasRightClick}
    >
      {units.map((unit) => (
        <Unit
          key={unit.id}
          unit={unit}
          onSelect={handleUnitSelect}
          onRightClick={handleUnitRightClick}
        />
      ))}

      {settings.showDamageNumbers &&
        damageEvents.map((event) => <DamageNumber key={event.id} event={event} />)}
    </div>
  );
};

export default GameCanvas;
