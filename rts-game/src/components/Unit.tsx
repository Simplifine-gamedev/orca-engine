import React from 'react';
import { Unit as UnitType } from '../types';
import { useGameStore } from '../store/gameStore';

interface UnitProps {
  unit: UnitType;
}

export const Unit: React.FC<UnitProps> = ({ unit }) => {
  const resources = useGameStore((state) => state.resources);

  const getUnitColor = () => {
    if (unit.isGathering) {
      return 'bg-yellow-600 border-yellow-400';
    }
    return unit.type === 'worker' ? 'bg-blue-400 border-blue-600' : 'bg-red-400 border-red-600';
  };

  const getUnitIcon = () => {
    if (unit.isGathering) return '⛏️';
    return unit.type === 'worker' ? '👷' : '⚔️';
  };

  const targetResource = unit.targetResource ? resources.get(unit.targetResource) : null;

  return (
    <div className="relative">
      <div
        className={`w-12 h-12 ${getUnitColor()} rounded-full flex items-center justify-center text-white shadow-lg border-2`}
        style={{
          position: 'absolute',
          left: unit.position.x,
          top: unit.position.y,
        }}
      >
        <div className="text-xl">{getUnitIcon()}</div>
      </div>
      
      {unit.isGathering && targetResource && (
        <div
          className="absolute bg-yellow-100 border border-yellow-500 rounded px-2 py-0.5 text-xs font-semibold whitespace-nowrap"
          style={{
            left: unit.position.x + 50,
            top: unit.position.y,
          }}
        >
          Gathering {targetResource.type}
        </div>
      )}
    </div>
  );
};

export default Unit;
