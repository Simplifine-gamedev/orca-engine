import React, { useCallback } from 'react';
import { useGameStore } from '../store/gameStore';

export interface BuildingProps {
  buildingId: string;
  unitTypes: string[];
}

export const Building: React.FC<BuildingProps> = ({ buildingId, unitTypes }) => {
  const { buildings, trainUnit, selectBuilding, selectedBuildingId } = useGameStore();
  const building = buildings.get(buildingId);

  const handleTrainUnit = useCallback(
    (unitType: string, event: React.MouseEvent) => {
      // Detect SHIFT key - queue 5 units instead of 1
      const count = event.shiftKey ? 5 : 1;
      trainUnit(buildingId, unitType, count);
      
      // Show visual feedback
      if (event.shiftKey) {
        console.log(`Queued ${count}x ${unitType}`);
      }
    },
    [buildingId, trainUnit]
  );

  const handleBuildingClick = useCallback(() => {
    selectBuilding(buildingId);
  }, [buildingId, selectBuilding]);

  if (!building) return null;

  const isSelected = selectedBuildingId === buildingId;
  const queueCount = building.unitQueue.length;

  return (
    <div
      className={`building ${isSelected ? 'selected' : ''}`}
      onClick={handleBuildingClick}
      style={{
        position: 'absolute',
        left: building.position.x,
        top: building.position.y,
        width: '100px',
        height: '100px',
        border: isSelected ? '3px solid #4CAF50' : '2px solid #888',
        borderRadius: '8px',
        backgroundColor: '#333',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
      }}
    >
      <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{building.type}</div>
      {queueCount > 0 && (
        <div
          style={{
            position: 'absolute',
            top: '-10px',
            right: '-10px',
            backgroundColor: '#2196F3',
            borderRadius: '50%',
            width: '24px',
            height: '24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '12px',
            fontWeight: 'bold',
            border: '2px solid white',
          }}
        >
          {queueCount}
        </div>
      )}
      {isSelected && (
        <div
          style={{
            marginTop: '10px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}
        >
          {unitTypes.map((unitType) => (
            <button
              key={unitType}
              onClick={(e) => {
                e.stopPropagation();
                handleTrainUnit(unitType, e);
              }}
              style={{
                padding: '4px 8px',
                fontSize: '10px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              title={`Click to train 1 ${unitType}. SHIFT+Click to train 5.`}
            >
              Train {unitType}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
