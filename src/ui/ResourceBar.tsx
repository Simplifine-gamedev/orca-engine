import * as React from 'react';
import { gameStore } from '../store/gameStore';
import { IdleWorkerButton } from './IdleWorkerButton';

export const ResourceBar: React.FC = () => {
  const [resources, setResources] = React.useState(gameStore.getState().resources);

  React.useEffect(() => {
    const updateResources = () => {
      setResources(gameStore.getState().resources);
    };

    const unsubscribe = gameStore.subscribe(updateResources);

    return unsubscribe;
  }, []);

  return (
    <div
      className="resource-bar"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        padding: '8px 16px',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        borderRadius: '4px',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      {/* Resources */}
      <div style={{ display: 'flex', gap: '16px', flex: 1 }}>
        <ResourceItem icon="🪵" label="Wood" value={resources.wood} color="#8b4513" />
        <ResourceItem icon="🌾" label="Food" value={resources.food} color="#22c55e" />
        <ResourceItem icon="🪙" label="Gold" value={resources.gold} color="#fbbf24" />
        <ResourceItem icon="🪨" label="Stone" value={resources.stone} color="#9ca3af" />
      </div>

      {/* Idle Worker Button */}
      <IdleWorkerButton />
    </div>
  );
};

interface ResourceItemProps {
  icon: string;
  label: string;
  value: number;
  color: string;
}

const ResourceItem: React.FC<ResourceItemProps> = ({ icon, label, value, color }) => {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        color: 'white',
      }}
    >
      <span style={{ fontSize: '20px' }}>{icon}</span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        <span style={{ fontSize: '10px', opacity: 0.7 }}>{label}</span>
        <span style={{ fontSize: '14px', fontWeight: 'bold', color }}>{value}</span>
      </div>
    </div>
  );
};
