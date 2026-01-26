import React, { useState } from 'react';
import { getFaction, getBuildingUnits, getUnitPreview } from '../config/factions';

export interface BuildingProps {
  buildingId: string;
  buildingType: string;
  playerFactionId: string; // CRITICAL: Must use player's faction, not hardcoded
  onUnitSpawn: (unitId: string) => void;
}

export const Building: React.FC<BuildingProps> = ({ 
  buildingId, 
  buildingType, 
  playerFactionId, 
  onUnitSpawn 
}) => {
  const [isTraining, setIsTraining] = useState(false);
  const [selectedUnit, setSelectedUnit] = useState<string | null>(null);

  // BUG FIX: Use playerFactionId instead of hardcoded 'dwarf'
  // BEFORE (BUGGY): const faction = getFaction('dwarf'); 
  // This would always show dwarf previews regardless of player faction
  const faction = getFaction(playerFactionId);
  
  if (!faction) {
    console.error(`Invalid faction ID: ${playerFactionId}`);
    return <div>Error: Invalid faction</div>;
  }

  // Get available units for this building type in the player's faction
  const availableUnitIds = getBuildingUnits(playerFactionId, buildingType);
  
  const handleTrainUnit = (unitId: string) => {
    setIsTraining(true);
    setSelectedUnit(unitId);
    
    // Simulate training time
    setTimeout(() => {
      onUnitSpawn(unitId);
      setIsTraining(false);
      setSelectedUnit(null);
    }, faction.units[unitId]?.buildTime * 1000 || 3000);
  };

  return (
    <div className="building-panel">
      <h3>{faction.buildings[buildingType]?.name || buildingType}</h3>
      <div className="unit-grid">
        {availableUnitIds.map((unitId) => {
          const unit = faction.units[unitId];
          if (!unit) return null;

          // CRITICAL: getUnitPreview uses playerFactionId to fetch correct preview
          const previewImage = getUnitPreview(playerFactionId, unitId);

          return (
            <div 
              key={unitId} 
              className={`unit-button ${selectedUnit === unitId ? 'training' : ''}`}
              onClick={() => !isTraining && handleTrainUnit(unitId)}
            >
              <img 
                src={previewImage} 
                alt={unit.name}
                className="unit-preview-image"
              />
              <div className="unit-info">
                <span className="unit-name">{unit.name}</span>
                <span className="unit-cost">Cost: {unit.cost}</span>
                <span className="unit-time">Time: {unit.buildTime}s</span>
              </div>
              {selectedUnit === unitId && (
                <div className="training-indicator">Training...</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Building;
