import { useGameStore } from './store/gameStore';
import { DamageNumber } from './effects/DamageNumber';
import { GameCanvas } from './components/GameCanvas';
import { Settings } from './components/Settings';

function App() {
  const { damageEvents, removeDamageEvent } = useGameStore();

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#1a1a1a',
        padding: '40px 20px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      <div
        style={{
          maxWidth: '800px',
          margin: '0 auto',
        }}
      >
        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: '30px' }}>
          <h1
            style={{
              margin: '0 0 10px 0',
              color: '#fff',
              fontSize: '36px',
              fontWeight: 'bold',
            }}
          >
            Orca RTS
          </h1>
          <p style={{ margin: 0, color: '#aaa', fontSize: '16px' }}>
            Combat Damage Numbers Demo
          </p>
        </header>

        {/* Game Canvas */}
        <GameCanvas />

        {/* Settings Panel */}
        <Settings />

        {/* Info */}
        <div
          style={{
            marginTop: '20px',
            padding: '15px',
            backgroundColor: '#2a2a2a',
            border: '2px solid #444',
            borderRadius: '8px',
            color: '#aaa',
            fontSize: '14px',
          }}
        >
          <strong style={{ color: '#fff' }}>Feature:</strong> Floating damage
          numbers with different colors for different damage types.
          <br />
          <strong style={{ color: '#fff' }}>Usage:</strong> Click on your units
          (green) to attack enemies (red). Damage numbers will appear above the
          target.
        </div>

        {/* Damage Numbers Layer */}
        {damageEvents.map((event) => (
          <DamageNumber
            key={event.id}
            event={event}
            onComplete={removeDamageEvent}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
