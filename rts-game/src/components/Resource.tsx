import React from 'react';
import { Resource as ResourceType } from '../types';

interface ResourceProps {
  resource: ResourceType;
}

export const Resource: React.FC<ResourceProps> = ({ resource }) => {
  const getResourceColor = () => {
    switch (resource.type) {
      case 'gold':
        return 'bg-yellow-500';
      case 'wood':
        return 'bg-amber-700';
      case 'stone':
        return 'bg-gray-400';
      default:
        return 'bg-gray-500';
    }
  };

  const getResourceIcon = () => {
    switch (resource.type) {
      case 'gold':
        return '⚱️';
      case 'wood':
        return '🌲';
      case 'stone':
        return '🪨';
      default:
        return '📦';
    }
  };

  return (
    <div
      className={`w-16 h-16 ${getResourceColor()} rounded-full flex flex-col items-center justify-center text-white shadow-lg border-4 border-opacity-50`}
      style={{
        position: 'absolute',
        left: resource.position.x,
        top: resource.position.y,
        borderColor: resource.type === 'gold' ? '#fbbf24' : '#9ca3af',
      }}
    >
      <div className="text-2xl">{getResourceIcon()}</div>
      <div className="text-xs font-bold">{resource.amount}</div>
    </div>
  );
};

export default Resource;
