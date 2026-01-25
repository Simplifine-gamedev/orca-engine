import React, { useRef, useEffect } from 'react';
import { Unit, SelectionBox } from '../types';
import { getSelectionColor } from '../utils/selection';

interface GameCanvasProps {
  units: Unit[];
  selectedUnitIds: string[];
  selectionBox: SelectionBox | null;
  onMouseDown: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  onMouseMove: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  onMouseUp: (e: React.MouseEvent<HTMLCanvasElement>) => void;
}

export const GameCanvas: React.FC<GameCanvasProps> = ({
  units,
  selectedUnitIds,
  selectionBox,
  onMouseDown,
  onMouseMove,
  onMouseUp,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw units
    units.forEach(unit => {
      const isSelected = selectedUnitIds.includes(unit.id);
      
      // Draw unit circle
      ctx.fillStyle = unit.color;
      ctx.beginPath();
      ctx.arc(unit.position.x, unit.position.y, unit.size, 0, Math.PI * 2);
      ctx.fill();

      // Draw selection ring
      if (isSelected) {
        ctx.strokeStyle = getSelectionColor(unit, true);
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(unit.position.x, unit.position.y, unit.size + 5, 0, Math.PI * 2);
        ctx.stroke();
        
        // Draw team indicator line underneath
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(unit.position.x - unit.size, unit.position.y + unit.size + 10);
        ctx.lineTo(unit.position.x + unit.size, unit.position.y + unit.size + 10);
        ctx.stroke();
      }

      // Draw health bar
      const barWidth = unit.size * 2;
      const barHeight = 4;
      const barX = unit.position.x - unit.size;
      const barY = unit.position.y - unit.size - 10;
      
      // Background
      ctx.fillStyle = '#333';
      ctx.fillRect(barX, barY, barWidth, barHeight);
      
      // Health
      const healthPercent = unit.health / unit.maxHealth;
      ctx.fillStyle = healthPercent > 0.5 ? '#0F0' : healthPercent > 0.25 ? '#FF0' : '#F00';
      ctx.fillRect(barX, barY, barWidth * healthPercent, barHeight);

      // Draw unit name
      ctx.fillStyle = '#FFF';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(unit.name, unit.position.x, unit.position.y + unit.size + 25);
    });

    // Draw selection box
    if (selectionBox) {
      const width = selectionBox.end.x - selectionBox.start.x;
      const height = selectionBox.end.y - selectionBox.start.y;
      
      // Determine box color based on what's being selected
      const unitsInBox = units.filter(unit => {
        const minX = Math.min(selectionBox.start.x, selectionBox.end.x);
        const maxX = Math.max(selectionBox.start.x, selectionBox.end.x);
        const minY = Math.min(selectionBox.start.y, selectionBox.end.y);
        const maxY = Math.max(selectionBox.start.y, selectionBox.end.y);
        
        return (
          unit.position.x >= minX &&
          unit.position.x <= maxX &&
          unit.position.y >= minY &&
          unit.position.y <= maxY
        );
      });
      
      const hasEnemy = unitsInBox.some(u => u.team === 'enemy');
      const hasFriendly = unitsInBox.some(u => u.team === 'friendly');
      
      let boxColor = 'rgba(0, 255, 0, 0.3)';
      let borderColor = '#00FF00';
      
      if (hasEnemy && hasFriendly) {
        boxColor = 'rgba(255, 255, 0, 0.3)';
        borderColor = '#FFFF00';
      } else if (hasEnemy) {
        boxColor = 'rgba(255, 0, 0, 0.3)';
        borderColor = '#FF0000';
      }
      
      // Fill
      ctx.fillStyle = boxColor;
      ctx.fillRect(selectionBox.start.x, selectionBox.start.y, width, height);
      
      // Border
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(selectionBox.start.x, selectionBox.start.y, width, height);
    }
  }, [units, selectedUnitIds, selectionBox]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={600}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      style={{ border: '2px solid #333', cursor: 'crosshair' }}
    />
  );
};
