import React, { useState } from 'react';
import { useGameStore } from '../store/gameStore';
import { Building, Position, LUMBER_CAMP_BONUS } from '../types';

interface LumberCampProps {
  onBuild?: (building: Building) => void;
}

const LumberCamp: React.FC<LumberCampProps> = ({ onBuild }) => {
  const [isBuilding, setIsBuilding] = useState(false);
  const buildings = useGameStore((state) => state.buildings);
  const addBuilding = useGameStore((state) => state.addBuilding);
  const removeBuilding = useGameStore((state) => state.removeBuilding);
  const spendResources = useGameStore((state) => state.spendResources);

  const LUMBER_CAMP_COST = {
    wood: 100,
    gold: 50,
  };

  const handleBuildLumberCamp = (position: Position) => {
    const canAfford = spendResources(LUMBER_CAMP_COST);
    
    if (!canAfford) {
      alert('Not enough resources! Need 100 wood and 50 gold.');
      return;
    }

    const newBuilding: Building = {
      id: `lumber_camp_${Date.now()}`,
      type: 'lumber_camp',
      position,
      gatheringBonus: LUMBER_CAMP_BONUS,
    };

    addBuilding(newBuilding);
    
    if (onBuild) {
      onBuild(newBuilding);
    }

    setIsBuilding(false);
  };

  const handlePlacementClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isBuilding) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    handleBuildLumberCamp({ x, y });
  };

  const lumberCamps = buildings.filter((b) => b.type === 'lumber_camp');

  return (
    <div className="lumber-camp-manager">
      <div 
        style={{
          padding: '15px',
          backgroundColor: '#1a1a1a',
          borderRadius: '8px',
          marginBottom: '15px',
        }}
      >
        <h3 style={{ color: '#fff', marginBottom: '10px', fontSize: '18px' }}>
          Lumber Camp
        </h3>
        
        <div style={{ color: '#ccc', fontSize: '14px', marginBottom: '10px' }}>
          <p>Build lumber camps near forests to increase wood gathering efficiency.</p>
          <p style={{ color: '#FFD700', marginTop: '5px' }}>
            Bonus: +{((LUMBER_CAMP_BONUS - 1) * 100).toFixed(0)}% gathering speed
          </p>
        </div>

        <div style={{ marginBottom: '10px' }}>
          <strong style={{ color: '#fff' }}>Cost:</strong>
          <span style={{ color: '#8B4513', marginLeft: '10px' }}>
            🪵 {LUMBER_CAMP_COST.wood} Wood
          </span>
          <span style={{ color: '#FFD700', marginLeft: '10px' }}>
            💰 {LUMBER_CAMP_COST.gold} Gold
          </span>
        </div>

        <button
          onClick={() => setIsBuilding(!isBuilding)}
          style={{
            padding: '10px 20px',
            backgroundColor: isBuilding ? '#f44336' : '#4CAF50',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
            width: '100%',
          }}
        >
          {isBuilding ? 'Cancel Building' : 'Build Lumber Camp'}
        </button>

        {isBuilding && (
          <div 
            style={{
              marginTop: '10px',
              padding: '10px',
              backgroundColor: '#2c2c2c',
              borderRadius: '5px',
              color: '#FFD700',
              fontSize: '12px',
              textAlign: 'center',
            }}
          >
            Click on the map to place the lumber camp
          </div>
        )}
      </div>

      {/* Lumber Camp List */}
      {lumberCamps.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h4 style={{ color: '#fff', marginBottom: '10px', fontSize: '16px' }}>
            Active Lumber Camps ({lumberCamps.length})
          </h4>
          {lumberCamps.map((camp) => (
            <div
              key={camp.id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px',
                backgroundColor: '#2c2c2c',
                borderRadius: '5px',
                marginBottom: '5px',
                color: '#fff',
              }}
            >
              <div>
                <div style={{ fontSize: '14px' }}>
                  🏗️ Lumber Camp
                </div>
                <div style={{ fontSize: '11px', color: '#888' }}>
                  Position: ({Math.floor(camp.position.x)}, {Math.floor(camp.position.y)})
                </div>
              </div>
              <button
                onClick={() => {
                  if (confirm('Destroy this lumber camp?')) {
                    removeBuilding(camp.id);
                  }
                }}
                style={{
                  padding: '5px 10px',
                  backgroundColor: '#f44336',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '12px',
                }}
              >
                Destroy
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Placement Overlay */}
      {isBuilding && (
        <div
          onClick={handlePlacementClick}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            cursor: 'crosshair',
            zIndex: 1000,
          }}
        />
      )}
    </div>
  );
};

export default LumberCamp;
