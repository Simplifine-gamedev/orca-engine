import * as React from 'react';
import { ResourceBar } from './ResourceBar';
import { gameStore } from '../store/gameStore';

export const BottomHUD: React.FC = () => {
  const [selectedUnits, setSelectedUnits] = React.useState<string[]>([]);

  React.useEffect(() => {
    const updateSelection = () => {
      setSelectedUnits(gameStore.getState().selectedUnits);
    };

    const unsubscribe = gameStore.subscribe(updateSelection);

    return unsubscribe;
  }, []);

  const selectedCount = selectedUnits.length;
  const state = gameStore.getState();
  const selectedUnitData = state.units.filter(u => selectedUnits.includes(u.id));

  return (
    <div
      className="bottom-hud"
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        padding: '16px',
        pointerEvents: 'none',
      }}
    >
      {/* Selected Units Info */}
      {selectedCount > 0 && (
        <div
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: '12px 16px',
            borderRadius: '4px',
            color: 'white',
            fontFamily: 'Arial, sans-serif',
            alignSelf: 'center',
            pointerEvents: 'auto',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
            {selectedCount} unit{selectedCount !== 1 ? 's' : ''} selected
          </div>
          <div style={{ fontSize: '12px', opacity: 0.8 }}>
            {selectedUnitData.map(u => u.type).join(', ')}
          </div>
        </div>
      )}

      {/* Resource Bar */}
      <div style={{ pointerEvents: 'auto', alignSelf: 'center' }}>
        <ResourceBar />
      </div>
    </div>
  );
};
