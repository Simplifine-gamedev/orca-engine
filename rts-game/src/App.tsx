import React from 'react';
import { ControlPoint } from './objects/ControlPoint';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>🗼 Orca RTS - Watchtower Vision System</h1>
        <p>Hover over watchtowers to see their vision radius</p>
      </header>

      <div className="game-map">
        {/* Neutral watchtower */}
        <ControlPoint
          id="watchtower-1"
          x={200}
          y={200}
          team="neutral"
          visionRadius={150}
          isWatchtower={true}
        />

        {/* Player-controlled watchtower */}
        <ControlPoint
          id="watchtower-2"
          x={500}
          y={300}
          team="player"
          visionRadius={180}
          isWatchtower={true}
        />

        {/* Enemy watchtower */}
        <ControlPoint
          id="watchtower-3"
          x={800}
          y={200}
          team="enemy"
          visionRadius={150}
          isWatchtower={true}
        />

        {/* Larger vision radius example */}
        <ControlPoint
          id="watchtower-4"
          x={350}
          y={500}
          team="player"
          visionRadius={220}
          isWatchtower={true}
        />
      </div>

      <div className="instructions">
        <h3>Features Implemented:</h3>
        <ul>
          <li>✅ Eye icon above watchtower (with floating animation)</li>
          <li>✅ Vision radius preview on hover (pulsing circle)</li>
          <li>✅ Tooltip explaining vision benefits</li>
          <li>✅ Team-based color coding (neutral/player/enemy)</li>
          <li>✅ Smooth animations and transitions</li>
        </ul>
      </div>
    </div>
  );
}

export default App;
