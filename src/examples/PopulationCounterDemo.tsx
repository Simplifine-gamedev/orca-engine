import React, { useEffect } from 'react';
import { ResourceBar } from '../ui/ResourceBar';
import { useGameStore } from '../store/gameStore';
import '../ui/ResourceBar.css';

/**
 * Demo showing the fixed population counter
 * 
 * Before Fix: Would show 100 (world population)
 * After Fix: Shows 25 (player's faction population only)
 */
export const PopulationCounterDemo: React.FC = () => {
  const addUnit = useGameStore(state => state.addUnit);
  const setPlayerFaction = useGameStore(state => state.setPlayerFaction);
  const playerPopulation = useGameStore(state => state.getPlayerPopulation());
  const worldPopulation = useGameStore(state => state.getWorldPopulation());
  
  useEffect(() => {
    // Initialize factions
    const initGame = () => {
      // Set player faction
      setPlayerFaction('player1');
      
      // Add 25 player units
      for (let i = 0; i < 25; i++) {
        addUnit({
          id: `player-unit-${i}`,
          type: 'warrior',
          factionId: 'player1',
          health: 100,
          maxHealth: 100
        });
      }
      
      // Add 75 enemy units (different factions)
      for (let i = 0; i < 75; i++) {
        const factionId = i < 35 ? 'enemy1' : 'enemy2';
        addUnit({
          id: `enemy-unit-${i}`,
          type: 'warrior',
          factionId: factionId,
          health: 100,
          maxHealth: 100
        });
      }
    };
    
    initGame();
  }, [addUnit, setPlayerFaction]);
  
  return (
    <div style={{ padding: '20px', backgroundColor: '#1a1a1a', minHeight: '100vh' }}>
      <h1 style={{ color: 'white', marginBottom: '20px' }}>
        Population Counter Fix Demo (ORC-138)
      </h1>
      
      {/* This is the fixed ResourceBar component */}
      <ResourceBar />
      
      <div style={{ marginTop: '40px', color: 'white', padding: '20px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}>
        <h2>Bug Fix Verification</h2>
        <div style={{ marginTop: '20px', display: 'grid', gap: '15px' }}>
          <div>
            <strong>Player Population:</strong> {playerPopulation} units
            <span style={{ marginLeft: '10px', color: '#00ff00' }}>✓ This is what shows in the UI</span>
          </div>
          <div>
            <strong>World Population:</strong> {worldPopulation} units
            <span style={{ marginLeft: '10px', color: '#ff6600' }}>✗ This was incorrectly shown before the fix</span>
          </div>
          <div style={{ marginTop: '10px', padding: '15px', backgroundColor: 'rgba(0,255,0,0.1)', borderLeft: '4px solid #00ff00' }}>
            <strong>Expected Behavior:</strong> Population counter should show 25 / 100 (player's units / max population)
          </div>
          <div style={{ padding: '15px', backgroundColor: 'rgba(255,0,0,0.1)', borderLeft: '4px solid #ff0000' }}>
            <strong>Bug Behavior:</strong> Was showing 100 (total world population across all factions)
          </div>
        </div>
        
        <div style={{ marginTop: '30px' }}>
          <h3>Test Scenario:</h3>
          <ul style={{ lineHeight: '1.8' }}>
            <li>Player faction has <strong>25 units</strong></li>
            <li>Enemy faction 1 has <strong>35 units</strong></li>
            <li>Enemy faction 2 has <strong>40 units</strong></li>
            <li>Total world population: <strong>100 units</strong></li>
            <li>The UI correctly displays: <strong>25 / 100</strong> (player units / max cap)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default PopulationCounterDemo;
