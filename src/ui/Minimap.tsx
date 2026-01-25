import React, { useEffect, useRef, useState } from 'react';

interface Unit {
  id: string;
  x: number;
  y: number;
  team: 'player' | 'enemy' | 'neutral';
  selected: boolean;
}

interface MinimapProps {
  units: Unit[];
  mapWidth: number;
  mapHeight: number;
  minimapSize?: number;
  onMinimapClick?: (x: number, y: number) => void;
}

export const Minimap: React.FC<MinimapProps> = ({
  units,
  mapWidth,
  mapHeight,
  minimapSize = 200,
  onMinimapClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hovering, setHovering] = useState(false);

  const scaleX = minimapSize / mapWidth;
  const scaleY = minimapSize / mapHeight;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, minimapSize, minimapSize);

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let i = 0; i <= minimapSize; i += gridSize) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, minimapSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(minimapSize, i);
      ctx.stroke();
    }

    // Draw units
    units.forEach((unit) => {
      const x = unit.x * scaleX;
      const y = unit.y * scaleY;

      // Determine unit color based on team
      let color = '#888';
      if (unit.team === 'player') {
        color = unit.selected ? '#00ff00' : '#00aa00';
      } else if (unit.team === 'enemy') {
        color = '#ff0000';
      } else {
        color = '#ffff00';
      }

      ctx.fillStyle = color;
      
      // Draw larger circle for selected units
      const radius = unit.selected ? 4 : 2;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Add a subtle glow for selected units
      if (unit.selected) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  }, [units, mapWidth, mapHeight, minimapSize, scaleX, scaleY]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onMinimapClick) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / minimapSize) * mapWidth;
    const y = ((e.clientY - rect.top) / minimapSize) * mapHeight;

    onMinimapClick(x, y);
  };

  return (
    <div
      className="minimap-container"
      style={{
        position: 'relative',
        width: minimapSize,
        height: minimapSize,
        border: '2px solid #444',
        borderRadius: '4px',
        overflow: 'hidden',
        cursor: hovering ? 'pointer' : 'default',
      }}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      <canvas
        ref={canvasRef}
        width={minimapSize}
        height={minimapSize}
        onClick={handleClick}
        style={{ display: 'block' }}
      />
      
      {/* Visual feedback for hover */}
      {hovering && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            border: '2px solid #00ff00',
            pointerEvents: 'none',
            borderRadius: '2px',
          }}
        />
      )}
    </div>
  );
};
