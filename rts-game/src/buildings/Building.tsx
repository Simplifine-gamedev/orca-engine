import React, { useState, useCallback } from 'react';
import { Building as BuildingType, Position } from '../types';
import { useGameStore } from '../store/gameStore';

interface BuildingProps {
  building: BuildingType;
  onRallyPointSet?: (position: Position) => void;
}

export const Building: React.FC<BuildingProps> = ({ building, onRallyPointSet }) => {
  const [isSettingRallyPoint, setIsSettingRallyPoint] = useState(false);
  const setRallyPoint = useGameStore((state) => state.setRallyPoint);
  const spawnUnit = useGameStore((state) => state.spawnUnit);

  const handleSetRallyPoint = useCallback(() => {
    setIsSettingRallyPoint(true);
  }, []);

  const handleMapClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!isSettingRallyPoint) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const position: Position = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

    setRallyPoint(building.id, position);
    setIsSettingRallyPoint(false);
    
    if (onRallyPointSet) {
      onRallyPointSet(position);
    }
  }, [isSettingRallyPoint, building.id, setRallyPoint, onRallyPointSet]);

  const handleSpawnWorker = useCallback(() => {
    spawnUnit(building.id, 'worker');
  }, [building.id, spawnUnit]);

  const getBuildingColor = () => {
    switch (building.type) {
      case 'town_hall':
        return 'bg-blue-600';
      case 'barracks':
        return 'bg-red-600';
      case 'farm':
        return 'bg-green-600';
      default:
        return 'bg-gray-600';
    }
  };

  const getRallyPointIndicatorColor = () => {
    if (!building.rallyPoint) return '';
    return building.rallyPoint.isResourceRallyPoint ? 'bg-yellow-400' : 'bg-white';
  };

  return (
    <div className="relative">
      <div
        className={`w-24 h-24 ${getBuildingColor()} rounded-lg flex flex-col items-center justify-center text-white shadow-lg`}
        style={{
          position: 'absolute',
          left: building.position.x,
          top: building.position.y,
        }}
      >
        <div className="text-sm font-bold">{building.type.replace('_', ' ')}</div>
        <div className="text-xs mt-1">ID: {building.id.slice(0, 6)}</div>
      </div>

      {building.rallyPoint && (
        <>
          <div
            className={`w-4 h-4 ${getRallyPointIndicatorColor()} rounded-full border-2 border-black animate-pulse`}
            style={{
              position: 'absolute',
              left: building.rallyPoint.position.x - 8,
              top: building.rallyPoint.position.y - 8,
            }}
          />
          
          <svg
            className="absolute pointer-events-none"
            style={{
              left: 0,
              top: 0,
              width: '100%',
              height: '100%',
            }}
          >
            <line
              x1={building.position.x + 48}
              y1={building.position.y + 48}
              x2={building.rallyPoint.position.x}
              y2={building.rallyPoint.position.y}
              stroke={building.rallyPoint.isResourceRallyPoint ? '#fbbf24' : '#ffffff'}
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.6"
            />
          </svg>

          {building.rallyPoint.isResourceRallyPoint && building.rallyPoint.targetResource && (
            <div
              className="absolute bg-yellow-100 border-2 border-yellow-500 rounded px-2 py-1 text-xs font-semibold"
              style={{
                left: building.rallyPoint.position.x + 10,
                top: building.rallyPoint.position.y - 10,
              }}
            >
              {building.rallyPoint.targetResource.type.toUpperCase()}
            </div>
          )}
        </>
      )}

      <div
        className="absolute top-0 left-32 bg-gray-800 text-white p-3 rounded-lg shadow-lg"
        style={{
          left: building.position.x + 100,
          top: building.position.y,
        }}
      >
        <h3 className="font-bold mb-2 text-sm">Building Controls</h3>
        <div className="flex flex-col gap-2">
          <button
            onClick={handleSetRallyPoint}
            className={`px-3 py-1 rounded text-xs ${
              isSettingRallyPoint
                ? 'bg-yellow-500 text-black'
                : 'bg-blue-500 hover:bg-blue-600'
            }`}
          >
            {isSettingRallyPoint ? 'Click on map...' : 'Set Rally Point'}
          </button>
          
          <button
            onClick={handleSpawnWorker}
            className="px-3 py-1 bg-green-500 hover:bg-green-600 rounded text-xs"
          >
            Spawn Worker
          </button>

          {building.rallyPoint && (
            <div className="text-xs mt-2 p-2 bg-gray-700 rounded">
              <div className="font-semibold">Rally Point:</div>
              <div>({Math.round(building.rallyPoint.position.x)}, {Math.round(building.rallyPoint.position.y)})</div>
              {building.rallyPoint.isResourceRallyPoint && (
                <div className="text-yellow-400 font-semibold mt-1">
                  Resource: {building.rallyPoint.targetResource?.type}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {isSettingRallyPoint && (
        <div
          className="fixed inset-0 cursor-crosshair"
          onClick={handleMapClick}
          style={{ zIndex: 1000 }}
        />
      )}
    </div>
  );
};

export default Building;
