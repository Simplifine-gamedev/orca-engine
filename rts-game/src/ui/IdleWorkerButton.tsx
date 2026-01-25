import React from 'react';
import { useGameStore } from '../store/gameStore';

interface IdleWorkerButtonProps {
  className?: string;
}

export const IdleWorkerButton: React.FC<IdleWorkerButtonProps> = ({ className = '' }) => {
  const idleWorkerCount = useGameStore((state) => state.getIdleWorkerCount());
  const selectAllIdleWorkers = useGameStore((state) => state.selectAllIdleWorkers);

  const handleClick = () => {
    if (idleWorkerCount > 0) {
      selectAllIdleWorkers();
    }
  };

  // Don't show button if no idle workers
  if (idleWorkerCount === 0) {
    return null;
  }

  return (
    <button
      className={`idle-worker-button ${className}`}
      onClick={handleClick}
      title="Select all idle workers (Period key)"
      aria-label={`Select ${idleWorkerCount} idle worker${idleWorkerCount > 1 ? 's' : ''}`}
    >
      <div className="button-content">
        <span className="worker-icon">👷</span>
        <span className="worker-count">{idleWorkerCount}</span>
        <span className="idle-indicator">💤</span>
      </div>
    </button>
  );
};
