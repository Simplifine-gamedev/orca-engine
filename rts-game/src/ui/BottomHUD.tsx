import React, { useEffect } from 'react';
import { ResourceBar } from './ResourceBar';
import { IdleWorkerButton } from './IdleWorkerButton';
import { useGameStore } from '../store/gameStore';

export const BottomHUD: React.FC = () => {
  const selectedUnits = useGameStore((state) => state.selectedUnits);
  const units = useGameStore((state) => state.units);
  const selectAllIdleWorkers = useGameStore((state) => state.selectAllIdleWorkers);

  // Set up hotkey listener for period key
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Period key (.) to select idle workers, like in Age of Empires
      if (event.key === '.' || event.key === 'Period') {
        event.preventDefault();
        selectAllIdleWorkers();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectAllIdleWorkers]);

  const getSelectedUnitsInfo = () => {
    if (selectedUnits.length === 0) {
      return 'No units selected';
    }

    const selectedUnitObjects = units.filter((unit) =>
      selectedUnits.includes(unit.id)
    );

    if (selectedUnits.length === 1) {
      const unit = selectedUnitObjects[0];
      return `${unit.type.charAt(0).toUpperCase() + unit.type.slice(1)} - ${unit.isIdle ? 'Idle' : unit.currentTask || 'Busy'}`;
    }

    return `${selectedUnits.length} units selected`;
  };

  return (
    <div className="bottom-hud">
      <div className="hud-left">
        <ResourceBar />
      </div>

      <div className="hud-center">
        <div className="unit-info">
          {getSelectedUnitsInfo()}
        </div>
      </div>

      <div className="hud-right">
        <IdleWorkerButton />
        <div className="hotkey-hint">
          Press <kbd>.</kbd> to select idle workers
        </div>
      </div>
    </div>
  );
};
