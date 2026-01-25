import React, { useRef, useEffect, useState, useCallback } from 'react';

// Entity types
interface Position {
  x: number;
  y: number;
}

interface Size {
  width: number;
  height: number;
}

interface Entity {
  id: string;
  type: 'unit' | 'building';
  position: Position;
  size: Size;
  playerId: number;
  selected: boolean;
  name: string;
}

interface MarqueeBox {
  start: Position;
  end: Position;
  active: boolean;
}

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 800;
const CURRENT_PLAYER_ID = 1;

// Initialize game entities
const createInitialEntities = (): Entity[] => {
  const entities: Entity[] = [];
  
  // Player 1 units (controllable)
  for (let i = 0; i < 5; i++) {
    entities.push({
      id: `unit-p1-${i}`,
      type: 'unit',
      position: { x: 100 + i * 60, y: 200 },
      size: { width: 30, height: 30 },
      playerId: 1,
      selected: false,
      name: `Worker ${i + 1}`
    });
  }
  
  // Player 1 buildings (controllable)
  entities.push({
    id: 'building-p1-1',
    type: 'building',
    position: { x: 150, y: 400 },
    size: { width: 80, height: 80 },
    playerId: 1,
    selected: false,
    name: 'Command Center'
  });
  
  entities.push({
    id: 'building-p1-2',
    type: 'building',
    position: { x: 300, y: 400 },
    size: { width: 60, height: 60 },
    playerId: 1,
    selected: false,
    name: 'Barracks'
  });
  
  entities.push({
    id: 'building-p1-3',
    type: 'building',
    position: { x: 450, y: 420 },
    size: { width: 50, height: 50 },
    playerId: 1,
    selected: false,
    name: 'Supply Depot'
  });
  
  // Player 2 units (enemy - should not be selectable)
  for (let i = 0; i < 3; i++) {
    entities.push({
      id: `unit-p2-${i}`,
      type: 'unit',
      position: { x: 800 + i * 60, y: 300 },
      size: { width: 30, height: 30 },
      playerId: 2,
      selected: false,
      name: `Enemy ${i + 1}`
    });
  }
  
  // Player 2 buildings (enemy - should not be selectable)
  entities.push({
    id: 'building-p2-1',
    type: 'building',
    position: { x: 900, y: 500 },
    size: { width: 80, height: 80 },
    playerId: 2,
    selected: false,
    name: 'Enemy Base'
  });
  
  return entities;
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [entities, setEntities] = useState<Entity[]>(createInitialEntities());
  const [marquee, setMarquee] = useState<MarqueeBox>({
    start: { x: 0, y: 0 },
    end: { x: 0, y: 0 },
    active: false
  });
  const [isDragging, setIsDragging] = useState(false);
  const [allowMixedSelection, setAllowMixedSelection] = useState(true);

  // Check if a point is inside an entity
  const isPointInEntity = (point: Position, entity: Entity): boolean => {
    return (
      point.x >= entity.position.x &&
      point.x <= entity.position.x + entity.size.width &&
      point.y >= entity.position.y &&
      point.y <= entity.position.y + entity.size.height
    );
  };

  // Check if entity intersects with marquee box
  const isEntityInMarquee = (entity: Entity, marqueeBox: MarqueeBox): boolean => {
    const minX = Math.min(marqueeBox.start.x, marqueeBox.end.x);
    const maxX = Math.max(marqueeBox.start.x, marqueeBox.end.x);
    const minY = Math.min(marqueeBox.start.y, marqueeBox.end.y);
    const maxY = Math.max(marqueeBox.start.y, marqueeBox.end.y);

    const entityRight = entity.position.x + entity.size.width;
    const entityBottom = entity.position.y + entity.size.height;

    // Check if rectangles intersect
    return !(
      entity.position.x > maxX ||
      entityRight < minX ||
      entity.position.y > maxY ||
      entityBottom < minY
    );
  };

  // Handle mouse down - start marquee selection
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on an entity first
    const clickedEntity = entities.find(entity => 
      entity.playerId === CURRENT_PLAYER_ID && isPointInEntity({ x, y }, entity)
    );

    if (clickedEntity && !e.shiftKey) {
      // Single click on entity - select only that entity
      setEntities(prev => prev.map(entity => ({
        ...entity,
        selected: entity.id === clickedEntity.id
      })));
    } else if (clickedEntity && e.shiftKey) {
      // Shift+click - toggle entity selection
      setEntities(prev => prev.map(entity => 
        entity.id === clickedEntity.id
          ? { ...entity, selected: !entity.selected }
          : entity
      ));
    } else {
      // Start marquee selection
      setIsDragging(true);
      setMarquee({
        start: { x, y },
        end: { x, y },
        active: true
      });

      // Clear selection if not holding shift
      if (!e.shiftKey) {
        setEntities(prev => prev.map(entity => ({ ...entity, selected: false })));
      }
    }
  }, [entities]);

  // Handle mouse move - update marquee box
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMarquee(prev => ({
      ...prev,
      end: { x, y }
    }));
  }, [isDragging]);

  // Handle mouse up - complete marquee selection
  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    setIsDragging(false);

    // Select entities within marquee
    setEntities(prev => {
      const newEntities = [...prev];
      const selectedTypes = new Set<string>();
      
      // First pass: identify what types are being selected
      newEntities.forEach(entity => {
        if (
          entity.playerId === CURRENT_PLAYER_ID &&
          isEntityInMarquee(entity, marquee)
        ) {
          selectedTypes.add(entity.type);
        }
      });

      // Check if mixed selection is happening
      const isMixedSelection = selectedTypes.size > 1;

      // Second pass: apply selection based on mixed selection setting
      return newEntities.map(entity => {
        if (
          entity.playerId === CURRENT_PLAYER_ID &&
          isEntityInMarquee(entity, marquee)
        ) {
          // If mixed selection not allowed and this is a mixed selection,
          // only select the first type encountered
          if (!allowMixedSelection && isMixedSelection) {
            const firstType = Array.from(selectedTypes)[0];
            if (entity.type === firstType) {
              return { ...entity, selected: true };
            }
          } else {
            return { ...entity, selected: true };
          }
        }
        return entity;
      });
    });

    setMarquee(prev => ({ ...prev, active: false }));
  }, [isDragging, marquee, allowMixedSelection]);

  // Render game on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw grid
    ctx.strokeStyle = '#2a2a4e';
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
      const isPlayerOwned = entity.playerId === CURRENT_PLAYER_ID;
      
      // Entity body
      if (entity.type === 'unit') {
        // Draw units as circles
        ctx.beginPath();
        ctx.arc(
          entity.position.x + entity.size.width / 2,
          entity.position.y + entity.size.height / 2,
          entity.size.width / 2,
          0,
          Math.PI * 2
        );
        ctx.fillStyle = isPlayerOwned ? '#4CAF50' : '#F44336';
        ctx.fill();
        
        if (entity.selected) {
          ctx.strokeStyle = '#FFD700';
          ctx.lineWidth = 3;
          ctx.stroke();
        } else {
          ctx.strokeStyle = isPlayerOwned ? '#2E7D32' : '#C62828';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      } else {
        // Draw buildings as rectangles
        ctx.fillStyle = isPlayerOwned ? '#2196F3' : '#FF5722';
        ctx.fillRect(
          entity.position.x,
          entity.position.y,
          entity.size.width,
          entity.size.height
        );
        
        if (entity.selected) {
          ctx.strokeStyle = '#FFD700';
          ctx.lineWidth = 4;
        } else {
          ctx.strokeStyle = isPlayerOwned ? '#1565C0' : '#D84315';
          ctx.lineWidth = 2;
        }
        ctx.strokeRect(
          entity.position.x,
          entity.position.y,
          entity.size.width,
          entity.size.height
        );

        // Draw building icon/text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(
          entity.type === 'building' ? '🏢' : '',
          entity.position.x + entity.size.width / 2,
          entity.position.y + entity.size.height / 2 + 5
        );
      }

      // Draw entity name on hover
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        entity.name,
        entity.position.x + entity.size.width / 2,
        entity.position.y - 5
      );
    });

    // Draw marquee selection box
    if (marquee.active) {
      const minX = Math.min(marquee.start.x, marquee.end.x);
      const minY = Math.min(marquee.start.y, marquee.end.y);
      const width = Math.abs(marquee.end.x - marquee.start.x);
      const height = Math.abs(marquee.end.y - marquee.start.y);

      // Fill
      ctx.fillStyle = 'rgba(100, 200, 100, 0.2)';
      ctx.fillRect(minX, minY, width, height);

      // Border
      ctx.strokeStyle = '#64C864';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(minX, minY, width, height);
      ctx.setLineDash([]);
    }
  }, [entities, marquee]);

  // Get selected entities info
  const selectedEntities = entities.filter(e => e.selected);
  const selectedUnits = selectedEntities.filter(e => e.type === 'unit');
  const selectedBuildings = selectedEntities.filter(e => e.type === 'building');

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '20px',
      backgroundColor: '#0f0f1e',
      minHeight: '100vh',
      color: '#FFFFFF'
    }}>
      <h1 style={{ marginBottom: '10px' }}>Orca RTS - Marquee Selection Demo</h1>
      
      <div style={{
        display: 'flex',
        gap: '20px',
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#1a1a2e',
        borderRadius: '8px',
        width: CANVAS_WIDTH
      }}>
        <div>
          <h3 style={{ margin: '0 0 10px 0' }}>Controls</h3>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
            <li>Click to select single unit/building</li>
            <li>Drag to create selection box (marquee)</li>
            <li>Shift+Click to toggle selection</li>
            <li>Only player-owned entities can be selected</li>
          </ul>
        </div>
        <div>
          <h3 style={{ margin: '0 0 10px 0' }}>Options</h3>
          <label style={{ display: 'flex', alignItems: 'center', fontSize: '14px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={allowMixedSelection}
              onChange={(e) => setAllowMixedSelection(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            Allow mixed selection (units + buildings)
          </label>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          border: '2px solid #4CAF50',
          cursor: isDragging ? 'crosshair' : 'default',
          backgroundColor: '#1a1a2e'
        }}
      />

      <div style={{
        marginTop: '20px',
        padding: '15px',
        backgroundColor: '#1a1a2e',
        borderRadius: '8px',
        width: CANVAS_WIDTH
      }}>
        <h3 style={{ margin: '0 0 10px 0' }}>Selection Info</h3>
        <div style={{ fontSize: '14px' }}>
          <p style={{ margin: '5px 0' }}>
            <strong>Total Selected:</strong> {selectedEntities.length}
          </p>
          <p style={{ margin: '5px 0' }}>
            <strong>Units:</strong> {selectedUnits.length} | <strong>Buildings:</strong> {selectedBuildings.length}
          </p>
          {selectedEntities.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <strong>Selected entities:</strong>
              <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
                {selectedEntities.map(entity => (
                  <li key={entity.id}>
                    {entity.name} ({entity.type})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <div style={{
        marginTop: '20px',
        padding: '15px',
        backgroundColor: '#1a1a2e',
        borderRadius: '8px',
        width: CANVAS_WIDTH,
        fontSize: '12px',
        color: '#888'
      }}>
        <h4 style={{ margin: '0 0 10px 0' }}>Implementation Notes</h4>
        <ul style={{ margin: 0, paddingLeft: '20px' }}>
          <li>✅ Marquee selection box for dragging</li>
          <li>✅ Selection works for both units and buildings</li>
          <li>✅ Only player-owned entities are selectable (player ID filtering)</li>
          <li>✅ Mixed selection toggle (can select units + buildings together or separately)</li>
          <li>✅ Visual feedback with selection highlights</li>
          <li>✅ Shift+Click for multi-selection</li>
        </ul>
      </div>
    </div>
  );
}
