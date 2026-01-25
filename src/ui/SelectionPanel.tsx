import React from 'react';
import { useGameStore } from '../store/gameStore';

export const SelectionPanel: React.FC = () => {
  const { buildings, selectedBuildingId, cancelUnit } = useGameStore();
  
  const selectedBuilding = selectedBuildingId
    ? buildings.get(selectedBuildingId)
    : null;

  if (!selectedBuilding) {
    return (
      <div
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: '#222',
          color: 'white',
          padding: '20px',
          borderRadius: '8px',
          minWidth: '300px',
          border: '2px solid #555',
        }}
      >
        <div style={{ textAlign: 'center', color: '#888' }}>
          No building selected
        </div>
      </div>
    );
  }

  const handleCancelUnit = (unitId: string) => {
    cancelUnit(selectedBuilding.id, unitId);
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: '#222',
        color: 'white',
        padding: '20px',
        borderRadius: '8px',
        minWidth: '400px',
        maxWidth: '600px',
        border: '2px solid #555',
      }}
    >
      <div style={{ marginBottom: '15px' }}>
        <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>
          {selectedBuilding.type}
        </h3>
        <div style={{ fontSize: '12px', color: '#aaa' }}>
          Position: ({selectedBuilding.position.x}, {selectedBuilding.position.y})
        </div>
      </div>

      <div>
        <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#4CAF50' }}>
          Training Queue ({selectedBuilding.unitQueue.length})
        </h4>
        
        {selectedBuilding.unitQueue.length === 0 ? (
          <div style={{ fontSize: '12px', color: '#888', fontStyle: 'italic' }}>
            No units in queue. Hold SHIFT while clicking train to queue 5 units.
          </div>
        ) : (
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            {selectedBuilding.unitQueue.map((unit, index) => (
              <div
                key={unit.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px',
                  marginBottom: '4px',
                  backgroundColor: index === 0 ? '#2a4a2a' : '#333',
                  borderRadius: '4px',
                  border: index === 0 ? '1px solid #4CAF50' : '1px solid #444',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1 }}>
                  <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#aaa' }}>
                    #{index + 1}
                  </div>
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
                      {unit.unitType}
                    </div>
                    {index === 0 && (
                      <div style={{ fontSize: '10px', color: '#4CAF50' }}>
                        Training... {Math.round(unit.progress)}%
                      </div>
                    )}
                  </div>
                </div>
                
                {index === 0 && unit.progress > 0 && (
                  <div
                    style={{
                      width: '100px',
                      height: '8px',
                      backgroundColor: '#444',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      marginRight: '10px',
                    }}
                  >
                    <div
                      style={{
                        width: `${unit.progress}%`,
                        height: '100%',
                        backgroundColor: '#4CAF50',
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </div>
                )}
                
                <button
                  onClick={() => handleCancelUnit(unit.id)}
                  style={{
                    padding: '4px 8px',
                    fontSize: '10px',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                  title="Cancel this unit"
                >
                  Cancel
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div
        style={{
          marginTop: '15px',
          padding: '10px',
          backgroundColor: '#1a1a1a',
          borderRadius: '4px',
          fontSize: '11px',
          color: '#888',
        }}
      >
        <strong style={{ color: '#aaa' }}>Tip:</strong> Hold <kbd style={{ padding: '2px 6px', backgroundColor: '#333', borderRadius: '2px', color: '#fff' }}>SHIFT</kbd> while clicking train to queue 5 units at once.
      </div>
    </div>
  );
};
