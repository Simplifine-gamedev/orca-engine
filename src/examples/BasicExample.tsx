import React from 'react';
import { GameWorld } from '../components/GameWorld';

/**
 * Basic example showing how to use the GameWorld component
 * with default settings.
 */
export const BasicExample: React.FC = () => {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <h1 style={{ position: 'absolute', top: 0, left: 0, zIndex: 2000, color: 'white', padding: '10px' }}>
        Orca RTS - Mob Lair Demo
      </h1>
      <GameWorld serverUrl="http://localhost:3001" />
    </div>
  );
};

export default BasicExample;
