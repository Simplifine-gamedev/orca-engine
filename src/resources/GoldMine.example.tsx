import React, { useState } from 'react';
import { GoldMine } from './GoldMine';
import { GoldMineState, ResourceType } from './types';

/**
 * Example usage of the improved GoldMine component
 * 
 * This demonstrates the fixes for ORC-123:
 * - Larger, more readable font size (18px with bold weight)
 * - Visual progress bar for quick status assessment
 * - Detailed selection panel when mine is selected
 */
export const GoldMineExample: React.FC = () => {
  const [selectedMine, setSelectedMine] = useState<string | null>(null);
  const [mines, setMines] = useState<GoldMineState[]>([
    {
      id: 'mine-1',
      type: ResourceType.GOLD_MINE,
      position: { x: 100, y: 100 },
      goldRemaining: 15000,
      maxGold: 15000,
      harvestRate: 10,
      isSelected: false,
    },
    {
      id: 'mine-2',
      type: ResourceType.GOLD_MINE,
      position: { x: 300, y: 150 },
      goldRemaining: 7500,
      maxGold: 15000,
      harvestRate: 10,
      isSelected: false,
    },
    {
      id: 'mine-3',
      type: ResourceType.GOLD_MINE,
      position: { x: 500, y: 120 },
      goldRemaining: 2000,
      maxGold: 15000,
      harvestRate: 10,
      isSelected: false,
    },
    {
      id: 'mine-4',
      type: ResourceType.GOLD_MINE,
      position: { x: 200, y: 300 },
      goldRemaining: 0,
      maxGold: 15000,
      harvestRate: 0,
      isSelected: false,
    },
  ]);

  const handleMineClick = (mineId: string) => {
    setSelectedMine(mineId);
    setMines((prevMines) =>
      prevMines.map((mine) => ({
        ...mine,
        isSelected: mine.id === mineId,
      }))
    );
  };

  const handleHarvest = (mineId: string, amount: number) => {
    setMines((prevMines) =>
      prevMines.map((mine) =>
        mine.id === mineId
          ? {
              ...mine,
              goldRemaining: Math.max(0, mine.goldRemaining - amount),
            }
          : mine
      )
    );
  };

  return (
    <div style={{ width: '100%', height: '100vh', position: 'relative', backgroundColor: '#2a2a2a' }}>
      <div style={{ position: 'absolute', top: 20, left: 20, color: '#fff', zIndex: 1000 }}>
        <h2>Gold Mine Readability Demo (ORC-123)</h2>
        <p style={{ fontSize: '14px', color: '#aaa' }}>
          Click on mines to see the enhanced selection panel
        </p>
        <div style={{ marginTop: '10px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '8px' }}>Improvements:</h3>
          <ul style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.6' }}>
            <li>✓ Font size increased from 10px to 18px (80% larger)</li>
            <li>✓ Bold weight and text shadow for better contrast</li>
            <li>✓ Visual progress bar with color coding</li>
            <li>✓ Detailed selection panel with large, readable stats</li>
            <li>✓ Warning indicators for nearly depleted mines</li>
          </ul>
        </div>
      </div>

      {/* Game view area */}
      <div style={{ position: 'relative', paddingTop: '220px' }}>
        {mines.map((mine) => (
          <div
            key={mine.id}
            onClick={() => handleMineClick(mine.id)}
            style={{ cursor: 'pointer' }}
          >
            <GoldMine
              goldRemaining={mine.goldRemaining}
              maxGold={mine.maxGold}
              position={mine.position}
              isSelected={mine.isSelected}
            />
          </div>
        ))}
      </div>

      {/* Control panel for demo purposes */}
      <div
        style={{
          position: 'fixed',
          bottom: 20,
          left: 20,
          backgroundColor: 'rgba(20, 20, 30, 0.95)',
          border: '2px solid #666',
          borderRadius: '8px',
          padding: '15px',
          color: '#fff',
        }}
      >
        <h4 style={{ margin: '0 0 10px 0' }}>Demo Controls</h4>
        {selectedMine && (
          <button
            onClick={() => handleHarvest(selectedMine, 500)}
            style={{
              padding: '8px 16px',
              backgroundColor: '#ffd700',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            Harvest 500 Gold
          </button>
        )}
        {!selectedMine && (
          <p style={{ fontSize: '13px', color: '#aaa', margin: 0 }}>
            Select a mine first
          </p>
        )}
      </div>
    </div>
  );
};

export default GoldMineExample;
