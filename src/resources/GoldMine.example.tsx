import React, { useState } from 'react';
import { GoldMine } from './GoldMine';

/**
 * Example usage of the GoldMine component in an RTS game
 */
export const GoldMineExample: React.FC = () => {
  const [selectedMineId, setSelectedMineId] = useState<number | null>(null);
  
  // Example gold mines on the map
  const goldMines = [
    {
      id: 1,
      remainingGold: 15000,
      maxGold: 20000,
      position: { x: 100, y: 150 }
    },
    {
      id: 2,
      remainingGold: 5000,
      maxGold: 20000,
      position: { x: 400, y: 200 }
    },
    {
      id: 3,
      remainingGold: 800,
      maxGold: 20000,
      position: { x: 700, y: 350 }
    }
  ];

  return (
    <div 
      className="game-map" 
      style={{ 
        position: 'relative', 
        width: '1024px', 
        height: '768px',
        backgroundColor: '#2a5a2a',
        border: '2px solid #333'
      }}
    >
      <h2 style={{ 
        position: 'absolute', 
        top: '10px', 
        left: '10px',
        color: '#fff',
        textShadow: '2px 2px 4px rgba(0,0,0,0.8)'
      }}>
        RTS Game - Gold Mines Example
      </h2>

      {goldMines.map((mine) => (
        <div 
          key={mine.id}
          onClick={() => setSelectedMineId(mine.id)}
        >
          <GoldMine
            remainingGold={mine.remainingGold}
            maxGold={mine.maxGold}
            position={mine.position}
            isSelected={selectedMineId === mine.id}
          />
        </div>
      ))}

      {/* Click anywhere to deselect */}
      <div 
        style={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0,
          zIndex: -1
        }}
        onClick={() => setSelectedMineId(null)}
      />
    </div>
  );
};

export default GoldMineExample;
