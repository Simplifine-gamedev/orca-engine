import React from 'react';
import useGameStore from '../store/gameStore';

const UI: React.FC = () => {
  const { settings, toggleDamageNumbers, initializeGame, units } = useGameStore();

  const uiContainerStyle: React.CSSProperties = {
    position: 'absolute',
    top: '10px',
    left: '10px',
    right: '10px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    pointerEvents: 'none',
    zIndex: 100,
  };

  const panelStyle: React.CSSProperties = {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: '15px',
    borderRadius: '8px',
    color: 'white',
    fontFamily: 'Arial, sans-serif',
    fontSize: '14px',
    pointerEvents: 'all',
  };

  const buttonStyle: React.CSSProperties = {
    padding: '8px 16px',
    margin: '5px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
  };

  const toggleButtonStyle: React.CSSProperties = {
    ...buttonStyle,
    backgroundColor: settings.showDamageNumbers ? '#4CAF50' : '#757575',
  };

  const infoStyle: React.CSSProperties = {
    marginTop: '10px',
    fontSize: '12px',
    opacity: 0.8,
  };

  const playerUnits = units.filter((u) => u.team === 'player').length;
  const enemyUnits = units.filter((u) => u.team === 'enemy').length;

  return (
    <div style={uiContainerStyle}>
      <div style={panelStyle}>
        <h2 style={{ margin: '0 0 10px 0', fontSize: '18px' }}>Orca RTS</h2>
        <div>
          <strong>Player Units:</strong> {playerUnits} | <strong>Enemy Units:</strong> {enemyUnits}
        </div>
        <div style={infoStyle}>
          <div>Left-click: Select unit</div>
          <div>Right-click: Move/Attack</div>
          <div>Space: Toggle damage numbers</div>
        </div>
      </div>

      <div style={panelStyle}>
        <button style={buttonStyle} onClick={initializeGame}>
          Reset Game
        </button>
        <button style={toggleButtonStyle} onClick={toggleDamageNumbers}>
          Damage Numbers: {settings.showDamageNumbers ? 'ON' : 'OFF'}
        </button>
      </div>
    </div>
  );
};

export default UI;
