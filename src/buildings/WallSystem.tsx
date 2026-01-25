import React, { useState, useEffect, useCallback, useRef } from 'react';

export interface WallSegment {
  id: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  cost: number;
  isValid: boolean;
}

export interface WallSystemProps {
  onWallPlaced?: (segment: WallSegment) => void;
  onCancelled?: () => void;
  resources: number;
  costPerUnit?: number;
  gridSize?: number;
}

export const WallSystem: React.FC<WallSystemProps> = ({
  onWallPlaced,
  onCancelled,
  resources,
  costPerUnit = 10,
  gridSize = 20,
}) => {
  const [isPlacing, setIsPlacing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);
  const [validTiles, setValidTiles] = useState<Set<string>>(new Set());
  const [hoveredTile, setHoveredTile] = useState<{ x: number; y: number } | null>(null);
  const [showTutorial, setShowTutorial] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hasShownTutorial = useRef(false);

  // Show tutorial on first wall build attempt
  useEffect(() => {
    const tutorialShown = localStorage.getItem('wall_tutorial_shown');
    if (!tutorialShown && !hasShownTutorial.current) {
      setShowTutorial(true);
      hasShownTutorial.current = true;
      localStorage.setItem('wall_tutorial_shown', 'true');
    }
  }, []);

  // Calculate valid placement areas
  useEffect(() => {
    // In a real game, this would check terrain, existing structures, etc.
    const valid = new Set<string>();
    for (let x = 0; x < 50; x++) {
      for (let y = 0; y < 50; y++) {
        // Example: only allow placement on grass tiles (simplified)
        if (!isObstacle(x, y)) {
          valid.add(`${x},${y}`);
        }
      }
    }
    setValidTiles(valid);
  }, []);

  const isObstacle = (x: number, y: number): boolean => {
    // Placeholder for obstacle detection
    // In real implementation, check against game state
    return false;
  };

  const snapToGrid = (x: number, y: number) => {
    return {
      x: Math.floor(x / gridSize) * gridSize,
      y: Math.floor(y / gridSize) * gridSize,
    };
  };

  const calculateWallCost = (start: { x: number; y: number }, end: { x: number; y: number }): number => {
    const distance = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
    const units = Math.ceil(distance / gridSize);
    return units * costPerUnit;
  };

  const isValidPlacement = (x: number, y: number): boolean => {
    const gridX = Math.floor(x / gridSize);
    const gridY = Math.floor(y / gridSize);
    return validTiles.has(`${gridX},${gridY}`);
  };

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const snapped = snapToGrid(x, y);

    if (!isValidPlacement(snapped.x, snapped.y)) {
      return;
    }

    if (!startPoint) {
      // First click - set start point
      setStartPoint(snapped);
      setIsPlacing(true);
    } else {
      // Second click - place wall
      const cost = calculateWallCost(startPoint, snapped);
      
      if (cost > resources) {
        alert('Insufficient resources!');
        return;
      }

      const segment: WallSegment = {
        id: `wall_${Date.now()}`,
        startX: startPoint.x,
        startY: startPoint.y,
        endX: snapped.x,
        endY: snapped.y,
        cost,
        isValid: true,
      };

      onWallPlaced?.(segment);
      setStartPoint(null);
      setCurrentPoint(null);
      setIsPlacing(false);
    }
  }, [startPoint, resources, onWallPlaced, costPerUnit, gridSize, validTiles]);

  // Right-click to cancel (much more intuitive than ESC)
  const handleRightClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    
    if (isPlacing || startPoint) {
      setStartPoint(null);
      setCurrentPoint(null);
      setIsPlacing(false);
      onCancelled?.();
    }
  }, [isPlacing, startPoint, onCancelled]);

  // Track mouse movement for preview
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const snapped = snapToGrid(x, y);

    setHoveredTile(snapped);
    
    if (startPoint) {
      setCurrentPoint(snapped);
    }
  }, [startPoint, gridSize]);

  // Draw the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Highlight valid placement areas
    ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
    validTiles.forEach(tile => {
      const [x, y] = tile.split(',').map(Number);
      ctx.fillRect(x * gridSize, y * gridSize, gridSize, gridSize);
    });

    // Highlight hovered tile
    if (hoveredTile) {
      const isValid = isValidPlacement(hoveredTile.x, hoveredTile.y);
      ctx.fillStyle = isValid ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 0, 0, 0.3)';
      ctx.fillRect(hoveredTile.x, hoveredTile.y, gridSize, gridSize);
    }

    // Draw start point
    if (startPoint) {
      ctx.fillStyle = 'rgba(0, 100, 255, 0.6)';
      ctx.fillRect(startPoint.x, startPoint.y, gridSize, gridSize);
      ctx.strokeStyle = '#0066ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(startPoint.x, startPoint.y, gridSize, gridSize);
    }

    // Draw preview line
    if (startPoint && currentPoint) {
      const cost = calculateWallCost(startPoint, currentPoint);
      const canAfford = cost <= resources;
      const isValid = isValidPlacement(currentPoint.x, currentPoint.y);

      ctx.strokeStyle = canAfford && isValid ? '#00ff00' : '#ff0000';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(startPoint.x + gridSize / 2, startPoint.y + gridSize / 2);
      ctx.lineTo(currentPoint.x + gridSize / 2, currentPoint.y + gridSize / 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw endpoint
      ctx.fillStyle = canAfford && isValid ? 'rgba(0, 255, 0, 0.4)' : 'rgba(255, 0, 0, 0.4)';
      ctx.fillRect(currentPoint.x, currentPoint.y, gridSize, gridSize);
    }
  }, [startPoint, currentPoint, hoveredTile, validTiles, resources, gridSize]);

  // Calculate preview cost for display
  const previewCost = startPoint && currentPoint 
    ? calculateWallCost(startPoint, currentPoint) 
    : 0;
  const canAfford = previewCost <= resources;

  return (
    <div className="wall-system-container" style={{ position: 'relative' }}>
      {/* Tutorial Tooltip */}
      {showTutorial && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '15px 20px',
            borderRadius: '8px',
            zIndex: 1000,
            maxWidth: '400px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
          }}
        >
          <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>Wall Building Tutorial</h3>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
            <li>Click once to set start point</li>
            <li>Click again to place wall</li>
            <li><strong>Right-click to cancel</strong> (no more ESC!)</li>
            <li>Green areas = valid placement</li>
            <li>Red preview = invalid or too expensive</li>
          </ul>
          <button
            onClick={() => setShowTutorial(false)}
            style={{
              marginTop: '10px',
              padding: '5px 15px',
              background: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Got it!
          </button>
        </div>
      )}

      {/* Cost Preview */}
      {startPoint && currentPoint && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: canAfford ? 'rgba(0, 200, 0, 0.9)' : 'rgba(200, 0, 0, 0.9)',
            color: 'white',
            padding: '10px 15px',
            borderRadius: '6px',
            fontWeight: 'bold',
            fontSize: '16px',
            zIndex: 999,
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
          }}
        >
          Cost: {previewCost} / {resources}
          {!canAfford && <div style={{ fontSize: '12px', marginTop: '4px' }}>Insufficient Resources!</div>}
        </div>
      )}

      {/* Status indicator */}
      {isPlacing && (
        <div
          style={{
            position: 'absolute',
            bottom: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 16px',
            borderRadius: '4px',
            fontSize: '14px',
            zIndex: 999,
          }}
        >
          Right-click to cancel
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        onClick={handleCanvasClick}
        onContextMenu={handleRightClick}
        onMouseMove={handleMouseMove}
        style={{
          border: '2px solid #333',
          cursor: isPlacing ? 'crosshair' : 'pointer',
          display: 'block',
        }}
      />
    </div>
  );
};

export default WallSystem;
