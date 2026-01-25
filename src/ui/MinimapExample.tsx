import React, { useState } from 'react';
import { Minimap } from './Minimap';

/**
 * Example usage of the Minimap component
 * Demonstrates how the minimap displays units without redundant count indicators
 */
export const MinimapExample: React.FC = () => {
  const [units, setUnits] = useState([
    { id: '1', x: 100, y: 150, team: 'player' as const, selected: true },
    { id: '2', x: 150, y: 180, team: 'player' as const, selected: true },
    { id: '3', x: 200, y: 250, team: 'player' as const, selected: false },
    { id: '4', x: 300, y: 350, team: 'enemy' as const, selected: false },
    { id: '5', x: 400, y: 450, team: 'enemy' as const, selected: false },
    { id: '6', x: 500, y: 300, team: 'neutral' as const, selected: false },
  ]);

  const selectedCount = units.filter(u => u.selected).length;

  const handleMinimapClick = (x: number, y: number) => {
    console.log(`Camera moved to: ${x.toFixed(0)}, ${y.toFixed(0)}`);
  };

  const handleSelectAll = () => {
    setUnits(units.map(u => ({ ...u, selected: u.team === 'player' })));
  };

  const handleDeselectAll = () => {
    setUnits(units.map(u => ({ ...u, selected: false })));
  };

  return (
    <div style={{ padding: '20px', backgroundColor: '#0a0a0a', minHeight: '100vh' }}>
      <h1 style={{ color: '#fff', marginBottom: '20px' }}>RTS Minimap Demo</h1>
      
      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
        <div>
          <Minimap
            units={units}
            mapWidth={1000}
            mapHeight={1000}
            minimapSize={300}
            onMinimapClick={handleMinimapClick}
          />
          
          {/* Selection info is shown separately, not on the minimap itself */}
          <div style={{ 
            marginTop: '10px', 
            padding: '10px', 
            backgroundColor: '#222', 
            borderRadius: '4px',
            color: '#fff',
            fontSize: '14px'
          }}>
            <strong>Selected Units:</strong> {selectedCount}
          </div>
        </div>

        <div style={{ color: '#fff' }}>
          <h2 style={{ fontSize: '18px', marginBottom: '10px' }}>Controls</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <button 
              onClick={handleSelectAll}
              style={{
                padding: '10px 20px',
                backgroundColor: '#00aa00',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Select All Player Units
            </button>
            <button 
              onClick={handleDeselectAll}
              style={{
                padding: '10px 20px',
                backgroundColor: '#aa0000',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Deselect All
            </button>
          </div>

          <div style={{ marginTop: '20px', fontSize: '14px', lineHeight: '1.6' }}>
            <h3 style={{ fontSize: '16px', marginBottom: '10px' }}>Legend:</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#00ff00' }} />
              <span>Selected Player Units (larger, bright green with glow)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#00aa00' }} />
              <span>Unselected Player Units (smaller, dark green)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '5px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ff0000' }} />
              <span>Enemy Units (red)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ffff00' }} />
              <span>Neutral Units (yellow)</span>
            </div>
          </div>

          <div style={{ 
            marginTop: '20px', 
            padding: '15px', 
            backgroundColor: '#1a3a1a', 
            borderRadius: '4px',
            fontSize: '14px'
          }}>
            <strong>✓ Fixed:</strong> The redundant selected units count has been removed from the minimap itself. 
            Selection is now indicated purely through visual cues (size, color, glow), making the UI cleaner and 
            less cluttered. The count is available separately in the UI where it's more appropriate.
          </div>
        </div>
      </div>
    </div>
  );
};
