import React, { useEffect, useRef, useState, useCallback } from 'react';
import './App.css';

// Types
interface Position {
  x: number;
  y: number;
}

interface Unit {
  id: string;
  type: 'unit';
  position: Position;
  size: number;
  playerId: number;
  selected: boolean;
}

interface Building {
  id: string;
  type: 'building';
  position: Position;
  width: number;
  height: number;
  playerId: number;
  buildingType: 'barracks' | 'factory' | 'headquarters';
  selected: boolean;
}

type Entity = Unit | Building;

interface MarqueeSelection {
  start: Position;
  end: Position;
  active: boolean;
}

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 800;
const PLAYER_ID = 1; // Current player

const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [marquee, setMarquee] = useState<MarqueeSelection>({
    start: { x: 0, y: 0 },
    end: { x: 0, y: 0 },
    active: false,
  });
  const [allowMixedSelection, setAllowMixedSelection] = useState(true);
  const [selectionInfo, setSelectionInfo] = useState<string>('');

  // Initialize game entities
  useEffect(() => {
    const initialEntities: Entity[] = [
      // Player units (player 1)
      { id: 'u1', type: 'unit', position: { x: 100, y: 100 }, size: 20, playerId: 1, selected: false },
      { id: 'u2', type: 'unit', position: { x: 150, y: 120 }, size: 20, playerId: 1, selected: false },
      { id: 'u3', type: 'unit', position: { x: 200, y: 140 }, size: 20, playerId: 1, selected: false },
      { id: 'u4', type: 'unit', position: { x: 250, y: 160 }, size: 20, playerId: 1, selected: false },
      
      // Enemy units (player 2)
      { id: 'u5', type: 'unit', position: { x: 900, y: 100 }, size: 20, playerId: 2, selected: false },
      { id: 'u6', type: 'unit', position: { x: 950, y: 120 }, size: 20, playerId: 2, selected: false },
      
      // Player buildings (player 1)
      { 
        id: 'b1', 
        type: 'building', 
        position: { x: 300, y: 300 }, 
        width: 80, 
        height: 80, 
        playerId: 1, 
        buildingType: 'headquarters',
        selected: false 
      },
      { 
        id: 'b2', 
        type: 'building', 
        position: { x: 500, y: 300 }, 
        width: 60, 
        height: 60, 
        playerId: 1, 
        buildingType: 'barracks',
        selected: false 
      },
      { 
        id: 'b3', 
        type: 'building', 
        position: { x: 700, y: 300 }, 
        width: 60, 
        height: 60, 
        playerId: 1, 
        buildingType: 'factory',
        selected: false 
      },
      
      // Enemy buildings (player 2)
      { 
        id: 'b4', 
        type: 'building', 
        position: { x: 900, y: 600 }, 
        width: 80, 
        height: 80, 
        playerId: 2, 
        buildingType: 'headquarters',
        selected: false 
      },
      { 
        id: 'b5', 
        type: 'building', 
        position: { x: 700, y: 600 }, 
        width: 60, 
        height: 60, 
        playerId: 2, 
        buildingType: 'barracks',
        selected: false 
      },
    ];
    
    setEntities(initialEntities);
  }, []);

  // Check if entity is within marquee selection
  const isEntityInMarquee = useCallback((entity: Entity, marqueeBox: { x: number; y: number; width: number; height: number }): boolean => {
    if (entity.type === 'unit') {
      const unitCenterX = entity.position.x;
      const unitCenterY = entity.position.y;
      const unitRadius = entity.size / 2;
      
      // Check if unit's center or any part overlaps with marquee
      return (
        unitCenterX + unitRadius >= marqueeBox.x &&
        unitCenterX - unitRadius <= marqueeBox.x + marqueeBox.width &&
        unitCenterY + unitRadius >= marqueeBox.y &&
        unitCenterY - unitRadius <= marqueeBox.y + marqueeBox.height
      );
    } else {
      // Building - check rectangle overlap
      return (
        entity.position.x < marqueeBox.x + marqueeBox.width &&
        entity.position.x + entity.width > marqueeBox.x &&
        entity.position.y < marqueeBox.y + marqueeBox.height &&
        entity.position.y + entity.height > marqueeBox.y
      );
    }
  }, []);

  // Handle mouse down - start marquee selection
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setMarquee({
      start: { x, y },
      end: { x, y },
      active: true,
    });
  }, []);

  // Handle mouse move - update marquee selection
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!marquee.active) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setMarquee(prev => ({
      ...prev,
      end: { x, y },
    }));
  }, [marquee.active]);

  // Handle mouse up - finalize selection
  const handleMouseUp = useCallback(() => {
    if (!marquee.active) return;
    
    const marqueeBox = {
      x: Math.min(marquee.start.x, marquee.end.x),
      y: Math.min(marquee.start.y, marquee.end.y),
      width: Math.abs(marquee.end.x - marquee.start.x),
      height: Math.abs(marquee.end.y - marquee.start.y),
    };
    
    // Update entity selection
    setEntities(prevEntities => {
      const entitiesInMarquee = prevEntities.filter(
        entity => isEntityInMarquee(entity, marqueeBox) && entity.playerId === PLAYER_ID
      );
      
      // Determine selection type
      let selectedUnits = 0;
      let selectedBuildings = 0;
      
      entitiesInMarquee.forEach(entity => {
        if (entity.type === 'unit') selectedUnits++;
        if (entity.type === 'building') selectedBuildings++;
      });
      
      // Apply mixed selection rules
      let shouldSelectUnits = selectedUnits > 0;
      let shouldSelectBuildings = selectedBuildings > 0;
      
      if (!allowMixedSelection && selectedUnits > 0 && selectedBuildings > 0) {
        // If mixed selection is disabled, prioritize units
        shouldSelectBuildings = false;
      }
      
      return prevEntities.map(entity => {
        const inMarquee = isEntityInMarquee(entity, marqueeBox);
        const isPlayerOwned = entity.playerId === PLAYER_ID;
        
        if (inMarquee && isPlayerOwned) {
          if (entity.type === 'unit' && shouldSelectUnits) {
            return { ...entity, selected: true };
          }
          if (entity.type === 'building' && shouldSelectBuildings) {
            return { ...entity, selected: true };
          }
        }
        
        return { ...entity, selected: false };
      });
    });
    
    setMarquee(prev => ({ ...prev, active: false }));
  }, [marquee, isEntityInMarquee, allowMixedSelection]);

  // Update selection info
  useEffect(() => {
    const selectedEntities = entities.filter(e => e.selected);
    const unitCount = selectedEntities.filter(e => e.type === 'unit').length;
    const buildingCount = selectedEntities.filter(e => e.type === 'building').length;
    
    if (selectedEntities.length === 0) {
      setSelectionInfo('No selection');
    } else {
      const parts = [];
      if (unitCount > 0) parts.push(`${unitCount} unit${unitCount > 1 ? 's' : ''}`);
      if (buildingCount > 0) parts.push(`${buildingCount} building${buildingCount > 1 ? 's' : ''}`);
      setSelectionInfo(`Selected: ${parts.join(', ')}`);
    }
  }, [entities]);

  // Render the game
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Draw grid
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    for (let x = 0; x < CANVAS_WIDTH; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, CANVAS_HEIGHT);
      ctx.stroke();
    }
    for (let y = 0; y < CANVAS_HEIGHT; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(CANVAS_WIDTH, y);
      ctx.stroke();
    }
    
    // Draw entities
    entities.forEach(entity => {
      if (entity.type === 'unit') {
        // Draw unit as circle
        ctx.beginPath();
        ctx.arc(entity.position.x, entity.position.y, entity.size / 2, 0, Math.PI * 2);
        ctx.fillStyle = entity.playerId === PLAYER_ID ? '#4a9eff' : '#ff4a4a';
        ctx.fill();
        
        // Draw selection indicator
        if (entity.selected) {
          ctx.strokeStyle = '#ffff00';
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        
        // Draw player indicator
        ctx.strokeStyle = entity.playerId === PLAYER_ID ? '#6ab7ff' : '#ff6a6a';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        // Draw building as rectangle
        ctx.fillStyle = entity.playerId === PLAYER_ID ? '#3d7a3d' : '#7a3d3d';
        ctx.fillRect(entity.position.x, entity.position.y, entity.width, entity.height);
        
        // Draw building border
        ctx.strokeStyle = entity.playerId === PLAYER_ID ? '#5a9a5a' : '#9a5a5a';
        ctx.lineWidth = 2;
        ctx.strokeRect(entity.position.x, entity.position.y, entity.width, entity.height);
        
        // Draw selection indicator
        if (entity.selected) {
          ctx.strokeStyle = '#ffff00';
          ctx.lineWidth = 4;
          ctx.strokeRect(entity.position.x - 2, entity.position.y - 2, entity.width + 4, entity.height + 4);
        }
        
        // Draw building type text
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(
          entity.buildingType.toUpperCase(),
          entity.position.x + entity.width / 2,
          entity.position.y + entity.height / 2
        );
      }
    });
    
    // Draw marquee selection box
    if (marquee.active) {
      const x = Math.min(marquee.start.x, marquee.end.x);
      const y = Math.min(marquee.start.y, marquee.end.y);
      const width = Math.abs(marquee.end.x - marquee.start.x);
      const height = Math.abs(marquee.end.y - marquee.start.y);
      
      // Draw semi-transparent fill
      ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
      ctx.fillRect(x, y, width, height);
      
      // Draw border
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);
    }
  }, [entities, marquee]);

  return (
    <div className="app">
      <div className="header">
        <h1>Orca RTS - Marquee Selection Demo</h1>
        <div className="controls">
          <label>
            <input
              type="checkbox"
              checked={allowMixedSelection}
              onChange={(e) => setAllowMixedSelection(e.target.checked)}
            />
            Allow mixed selection (units + buildings)
          </label>
        </div>
      </div>
      
      <div className="game-container">
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          style={{ border: '2px solid #333' }}
        />
      </div>
      
      <div className="info-panel">
        <div className="selection-info">{selectionInfo}</div>
        <div className="instructions">
          <h3>Instructions:</h3>
          <ul>
            <li>Click and drag to create a selection box</li>
            <li>Blue units and green buildings are yours (Player 1)</li>
            <li>Red units and buildings belong to the enemy (Player 2)</li>
            <li>Only player-owned entities can be selected</li>
            <li>Toggle mixed selection to allow/prevent selecting units and buildings together</li>
          </ul>
        </div>
        
        <div className="legend">
          <h3>Legend:</h3>
          <div className="legend-item">
            <div className="color-box" style={{ backgroundColor: '#4a9eff' }}></div>
            <span>Your Units</span>
          </div>
          <div className="legend-item">
            <div className="color-box" style={{ backgroundColor: '#3d7a3d' }}></div>
            <span>Your Buildings</span>
          </div>
          <div className="legend-item">
            <div className="color-box" style={{ backgroundColor: '#ff4a4a' }}></div>
            <span>Enemy Units</span>
          </div>
          <div className="legend-item">
            <div className="color-box" style={{ backgroundColor: '#7a3d3d' }}></div>
            <span>Enemy Buildings</span>
          </div>
          <div className="legend-item">
            <div className="color-box" style={{ backgroundColor: '#ffff00', width: '20px', height: '4px' }}></div>
            <span>Selected (Yellow)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
