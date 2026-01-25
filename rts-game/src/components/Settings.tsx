import { useGameStore } from '../store/gameStore';

export const Settings: React.FC = () => {
  const { settings, toggleDamageNumbers, updateSettings } = useGameStore();

  return (
    <div
      style={{
        marginTop: '20px',
        padding: '20px',
        backgroundColor: '#2a2a2a',
        border: '2px solid #444',
        borderRadius: '8px',
      }}
    >
      <h3 style={{ margin: '0 0 15px 0', color: '#fff' }}>Settings</h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        {/* Damage Numbers Toggle */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: '#fff',
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={settings.showDamageNumbers}
            onChange={toggleDamageNumbers}
            style={{
              width: '20px',
              height: '20px',
              cursor: 'pointer',
            }}
          />
          <span>Show Damage Numbers</span>
        </label>

        {/* Sound Toggle */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: '#fff',
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={settings.soundEnabled}
            onChange={(e) =>
              updateSettings({ soundEnabled: e.target.checked })
            }
            style={{
              width: '20px',
              height: '20px',
              cursor: 'pointer',
            }}
          />
          <span>Sound Effects</span>
        </label>

        {/* Music Volume */}
        <div style={{ color: '#fff' }}>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}
          >
            <span style={{ minWidth: '120px' }}>Music Volume:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={settings.musicVolume}
              onChange={(e) =>
                updateSettings({ musicVolume: parseFloat(e.target.value) })
              }
              style={{
                flex: 1,
                cursor: 'pointer',
              }}
            />
            <span style={{ minWidth: '40px', textAlign: 'right' }}>
              {Math.round(settings.musicVolume * 100)}%
            </span>
          </label>
        </div>
      </div>
    </div>
  );
};
