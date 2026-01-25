// THIS IS THE BUGGY VERSION - DO NOT USE
// This file shows how the bug manifested before the fix

import React, { useState } from 'react';
import { getFaction, getBuildingUnits, getUnitPreview } from './src/config/factions';

export interface BuildingProps {
  buildingId: string;
  buildingType: string;
  playerFactionId: string;
  onUnitSpawn: (unitId: string) => void;
}

export const BuggyBuilding: React.FC<BuildingProps> = ({ 
  buildingId, 
  buildingType, 
  playerFactionId, 
  onUnitSpawn 
}) => {
  const [isTraining, setIsTraining] = useState(false);
  const [selectedUnit, setSelectedUnit] = useState<string | null>(null);

  // 🐛 BUG: Hardcoded 'dwarf' faction instead of using playerFactionId
  // This causes ALL players to see dwarf unit previews, regardless of their faction
  const faction = getFaction('dwarf'); // ❌ WRONG!
  
  if (!faction) {
    console.error(`Invalid faction ID`);
    return <div>Error: Invalid faction</div>;
  }

  // 🐛 BUG: Using hardcoded 'dwarf' faction to get available units
  // This returns dwarf units (warrior, rifleman, hammerer)
  // even when player is human
  const availableUnitIds = getBuildingUnits('dwarf', buildingType); // ❌ WRONG!
  
  const handleTrainUnit = (unitId: string) => {
    setIsTraining(true);
    setSelectedUnit(unitId);
    
    // The actual spawning logic correctly uses playerFactionId
    // This is why the spawned units are correct, but previews are wrong!
    setTimeout(() => {
      onUnitSpawn(unitId); // This uses the correct faction internally
      setIsTraining(false);
      setSelectedUnit(null);
    }, faction.units[unitId]?.buildTime * 1000 || 3000);
  };

  return (
    <div className="building-panel">
      <h3>{faction.buildings[buildingType]?.name || buildingType}</h3>
      <div className="unit-grid">
        {availableUnitIds.map((unitId) => {
          // 🐛 Since we're using dwarf faction, this will be dwarf units
          const unit = faction.units[unitId];
          if (!unit) return null;

          // 🐛 This returns dwarf preview images even for human players
          const previewImage = getUnitPreview('dwarf', unitId); // ❌ WRONG!

          return (
            <div 
              key={unitId} 
              className={`unit-button ${selectedUnit === unitId ? 'training' : ''}`}
              onClick={() => !isTraining && handleTrainUnit(unitId)}
            >
              {/* Shows: warrior_preview.png, rifleman_preview.png, hammerer_preview.png */}
              {/* Should show: footman_preview.png, archer_preview.png, knight_preview.png */}
              <img 
                src={previewImage} 
                alt={unit.name}
                className="unit-preview-image"
              />
              <div className="unit-info">
                {/* Shows: "Dwarf Warrior", "Rifleman", "Hammerer" */}
                {/* Should show: "Footman", "Archer", "Knight" */}
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

/*
 * DEMONSTRATION OF THE BUG:
 * 
 * Scenario: Human player selects their barracks
 * 
 * playerFactionId = 'human'
 * 
 * What happens in buggy code:
 * 1. getFaction('dwarf') ❌ - Gets dwarf faction data
 * 2. getBuildingUnits('dwarf', 'barracks') ❌ - Gets dwarf units: [warrior, rifleman, hammerer]
 * 3. getUnitPreview('dwarf', 'warrior') ❌ - Gets dwarf preview: warrior_preview.png
 * 4. UI shows:
 *    - Preview: Dwarf Warrior image ❌
 *    - Name: "Dwarf Warrior" ❌
 *    - Button: Trains... Human Footman ✅ (onUnitSpawn has correct logic)
 * 
 * Result: Player sees dwarf previews but spawns human units!
 * 
 * User feedback: "human barracks is showing the previews of drawven units.... 
 *                 Weird bug. But it spawns the human units" - Haridzieko
 */

export default BuggyBuilding;
