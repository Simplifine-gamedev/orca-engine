/**
 * Building Component - Renders buildings with rally point indicators
 */

import React, { useState, useEffect } from 'react';
import { gameStore, Building as BuildingType, RallyPoint } from '../store/gameStore';

interface BuildingProps {
  building: BuildingType;
  onSelect: () => void;
  onSetRallyPoint: (position: { x: number; y: number }) => void;
}

export const Building: React.FC<BuildingProps> = ({ building, onSelect, onSetRallyPoint }) => {
  const [isSelected, setIsSelected] = useState(false);
  const [isSettingRallyPoint, setIsSettingRallyPoint] = useState(false);

  useEffect(() => {
    const unsubscribe = gameStore.subscribe(() => {
      const selectedBuilding = gameStore.getSelectedBuilding();
      setIsSelected(selectedBuilding?.id === building.id);
    });

    return unsubscribe;
  }, [building.id]);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect();
  };

  const handleSetRallyPoint = () => {
    setIsSettingRallyPoint(true);
  };

  const handleMapClick = (e: React.MouseEvent) => {
    if (isSettingRallyPoint) {
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const position = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
      onSetRallyPoint(position);
      setIsSettingRallyPoint(false);
    }
  };

  return (
    <div className="building-container" onClick={handleMapClick}>
      <div
        className={`building ${building.type} ${isSelected ? 'selected' : ''}`}
        style={{
          position: 'absolute',
          left: building.position.x,
          top: building.position.y,
          width: 80,
          height: 80,
          backgroundColor: building.type === 'townhall' ? '#8B4513' : '#696969',
          border: isSelected ? '3px solid yellow' : '2px solid black',
          borderRadius: 4,
          cursor: 'pointer',
        }}
        onClick={handleClick}
      >
        <div style={{ padding: 8, color: 'white', fontSize: 12 }}>
          {building.type}
        </div>
      </div>

      {/* Rally Point Indicator */}
      {building.rallyPoint && <RallyPointIndicator rallyPoint={building.rallyPoint} buildingPosition={building.position} />}

      {/* Rally Point Controls */}
      {isSelected && (
        <div
          className="building-controls"
          style={{
            position: 'absolute',
            left: building.position.x,
            top: building.position.y + 90,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 8,
            borderRadius: 4,
            color: 'white',
          }}
        >
          <button
            onClick={handleSetRallyPoint}
            style={{
              padding: '4px 8px',
              marginRight: 4,
              backgroundColor: isSettingRallyPoint ? '#ff6b6b' : '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: 3,
              cursor: 'pointer',
            }}
          >
            {isSettingRallyPoint ? 'Click to set rally point...' : 'Set Rally Point'}
          </button>
          <button
            onClick={() => gameStore.spawnUnit(building.id, 'worker')}
            style={{
              padding: '4px 8px',
              backgroundColor: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: 3,
              cursor: 'pointer',
            }}
          >
            Train Worker
          </button>
        </div>
      )}
    </div>
  );
};

interface RallyPointIndicatorProps {
  rallyPoint: RallyPoint;
  buildingPosition: { x: number; y: number };
}

const RallyPointIndicator: React.FC<RallyPointIndicatorProps> = ({ rallyPoint, buildingPosition }) => {
  const isOnResource = !!rallyPoint.targetResourceId;
  
  return (
    <>
      {/* Line from building to rally point */}
      <svg
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          pointerEvents: 'none',
          width: '100%',
          height: '100%',
        }}
      >
        <line
          x1={buildingPosition.x + 40}
          y1={buildingPosition.y + 40}
          x2={rallyPoint.position.x}
          y2={rallyPoint.position.y}
          stroke={isOnResource ? '#FFD700' : '#00FF00'}
          strokeWidth="2"
          strokeDasharray="5,5"
        />
      </svg>

      {/* Rally point flag/marker */}
      <div
        className="rally-point-marker"
        style={{
          position: 'absolute',
          left: rallyPoint.position.x - 10,
          top: rallyPoint.position.y - 20,
          width: 20,
          height: 20,
          pointerEvents: 'none',
        }}
      >
        {isOnResource ? (
          // Special indicator for resource rally point
          <div style={{ position: 'relative' }}>
            <div
              style={{
                width: 20,
                height: 20,
                backgroundColor: '#FFD700',
                border: '2px solid #FFA500',
                borderRadius: '50%',
                boxShadow: '0 0 10px rgba(255, 215, 0, 0.8)',
              }}
            />
            <div
              style={{
                position: 'absolute',
                top: -8,
                left: -8,
                fontSize: 24,
              }}
            >
              ⛏️
            </div>
          </div>
        ) : (
          // Regular rally point flag
          <div style={{ position: 'relative' }}>
            <div
              style={{
                width: 0,
                height: 0,
                borderLeft: '10px solid transparent',
                borderRight: '10px solid transparent',
                borderBottom: '20px solid #00FF00',
              }}
            />
            <div
              style={{
                position: 'absolute',
                width: 2,
                height: 20,
                backgroundColor: '#666',
                left: 9,
                top: 20,
              }}
            />
          </div>
        )}
      </div>

      {/* Resource indicator text */}
      {isOnResource && rallyPoint.targetResource && (
        <div
          style={{
            position: 'absolute',
            left: rallyPoint.position.x - 30,
            top: rallyPoint.position.y + 10,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: '#FFD700',
            padding: '2px 6px',
            borderRadius: 3,
            fontSize: 11,
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          Gathering {rallyPoint.targetResource.type}
        </div>
      )}
    </>
  );
};

export default Building;
