import * as React from 'react';
import { BottomHUD } from './ui/BottomHUD';
import { gameStore } from './store/gameStore';
import { useHotkeys } from './hooks/useHotkeys';
import { Unit, UnitType } from './types/game';

export const App: React.FC = () => {
  React.useEffect(() => {
    // Initialize with some test workers for demonstration
    const testWorkers: Unit[] = [
      {
        id: 'worker-1',
        position: { x: 100, y: 100 },
        type: UnitType.WORKER,
        isSelected: false,
        isIdle: true,
      },
      {
        id: 'worker-2',
        position: { x: 150, y: 100 },
        type: UnitType.WORKER,
        isSelected: false,
        isIdle: true,
      },
      {
        id: 'worker-3',
        position: { x: 200, y: 100 },
        type: UnitType.WORKER,
        isSelected: false,
        isIdle: false, // Busy worker
      },
      {
        id: 'worker-4',
        position: { x: 250, y: 100 },
        type: UnitType.WORKER,
        isSelected: false,
        isIdle: true,
      },
    ];

    testWorkers.forEach((worker) => gameStore.addUnit(worker));
  }, []);

  // Setup hotkeys
  useHotkeys([
    {
      key: '.',
      callback: () => {
        gameStore.selectIdleWorkers();
      },
      description: 'Select idle workers',
    },
    {
      key: 'Escape',
      callback: () => {
        gameStore.selectUnits([]);
      },
      description: 'Deselect all units',
    },
  ]);

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#1a1a1a',
        overflow: 'hidden',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      {/* Game viewport placeholder */}
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
        }}
      >
        <h1 style={{ fontSize: '48px', marginBottom: '16px' }}>Orca RTS</h1>
        <p style={{ fontSize: '18px', opacity: 0.7, marginBottom: '8px' }}>
          Press <kbd style={{ 
            padding: '4px 8px', 
            backgroundColor: 'rgba(255, 255, 255, 0.2)', 
            borderRadius: '4px',
            fontFamily: 'monospace'
          }}>.</kbd> (period) to select idle workers
        </p>
        <p style={{ fontSize: '14px', opacity: 0.5 }}>
          Click the orange button in the bottom bar to select idle workers
        </p>
      </div>

      {/* Bottom HUD with resource bar and idle worker button */}
      <BottomHUD />
    </div>
  );
};

export default App;
