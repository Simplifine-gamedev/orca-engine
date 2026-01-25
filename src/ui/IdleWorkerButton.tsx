import * as React from 'react';
import { gameStore } from '../store/gameStore';

interface IdleWorkerButtonProps {
  className?: string;
}

export const IdleWorkerButton: React.FC<IdleWorkerButtonProps> = ({ className = '' }) => {
  const [idleWorkerCount, setIdleWorkerCount] = React.useState(0);

  React.useEffect(() => {
    const updateCount = () => {
      setIdleWorkerCount(gameStore.getIdleWorkerCount());
    };

    // Initial count
    updateCount();

    // Subscribe to store changes
    const unsubscribe = gameStore.subscribe(updateCount);

    return unsubscribe;
  }, []);

  const handleClick = () => {
    gameStore.selectIdleWorkers();
  };

  // Don't render if no idle workers
  if (idleWorkerCount === 0) {
    return null;
  }

  return (
    <button
      onClick={handleClick}
      className={`idle-worker-button ${className}`}
      title="Select idle workers (Period key)"
      style={{
        padding: '8px 12px',
        backgroundColor: '#f59e0b',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        fontWeight: 'bold',
        fontSize: '14px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        transition: 'all 0.2s',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = '#d97706';
        e.currentTarget.style.transform = 'translateY(-1px)';
        e.currentTarget.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.3)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = '#f59e0b';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
      }}
    >
      {/* Worker icon */}
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="currentColor"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z" />
      </svg>
      
      {/* Count badge */}
      <span style={{ fontSize: '16px' }}>
        {idleWorkerCount}
      </span>
    </button>
  );
};
