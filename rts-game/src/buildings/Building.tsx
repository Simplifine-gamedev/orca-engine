import React from 'react';
import { useGameStore } from '../store/gameStore';
import { Building as BuildingType } from '../types';

interface BuildingProps {
  building: BuildingType;
}

export const Building: React.FC<BuildingProps> = ({ building }) => {
  const { units, releaseUnit, releaseAllUnits, selectBuilding, selectedBuildingId } = useGameStore();
  const isSelected = selectedBuildingId === building.id;

  const garrisonedUnits = building.garrisonedUnits
    .map((unitId) => units[unitId])
    .filter((unit) => unit !== undefined);

  const handleBuildingClick = () => {
    selectBuilding(building.id);
  };

  const handleReleaseUnit = (unitId: string) => {
    releaseUnit(unitId, building.id);
  };

  const handleReleaseAll = () => {
    releaseAllUnits(building.id);
  };

  return (
    <div
      onClick={handleBuildingClick}
      style={{
        position: 'absolute',
        left: building.position.x,
        top: building.position.y,
        width: 80,
        height: 80,
        backgroundColor: '#8B4513',
        border: isSelected ? '3px solid yellow' : '2px solid #654321',
        borderRadius: 4,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontWeight: 'bold',
        fontSize: 12,
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div>{building.name}</div>
        <div style={{ fontSize: 10 }}>
          {building.health}/{building.maxHealth}
        </div>
        {garrisonedUnits.length > 0 && (
          <div style={{ fontSize: 10, color: '#90EE90' }}>
            Units: {garrisonedUnits.length}
          </div>
        )}
      </div>

      {isSelected && garrisonedUnits.length > 0 && (
        <div
          style={{
            position: 'absolute',
            top: 90,
            left: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            padding: 10,
            borderRadius: 4,
            minWidth: 200,
            zIndex: 1000,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <h3 style={{ margin: '0 0 10px 0', fontSize: 14 }}>
            Garrisoned Units ({garrisonedUnits.length}/{building.maxGarrison})
          </h3>
          
          <div style={{ marginBottom: 10 }}>
            {garrisonedUnits.map((unit) => (
              <div
                key={unit.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 5,
                  padding: 5,
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 3,
                }}
              >
                <span style={{ fontSize: 12 }}>
                  {unit.name} ({unit.health}/{unit.maxHealth})
                </span>
                <button
                  onClick={() => handleReleaseUnit(unit.id)}
                  style={{
                    padding: '2px 8px',
                    fontSize: 10,
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: 3,
                    cursor: 'pointer',
                  }}
                >
                  Release
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={handleReleaseAll}
            style={{
              width: '100%',
              padding: '8px',
              fontSize: 12,
              backgroundColor: '#FF5722',
              color: 'white',
              border: 'none',
              borderRadius: 3,
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            Release All Units
          </button>
        </div>
      )}
    </div>
  );
};
