import React from 'react';
import Building from '../buildings/Building';
import { getFaction, getUnitPreview } from '../config/factions';

export interface SelectionPanelProps {
  selectedEntityId: string | null;
  selectedEntityType: 'unit' | 'building' | null;
  playerFactionId: string;
  entities: {
    [id: string]: {
      id: string;
      type: string;
      factionId: string; // Each entity stores its faction
      health: number;
      maxHealth: number;
    };
  };
  onUnitSpawn: (unitId: string) => void;
}

export const SelectionPanel: React.FC<SelectionPanelProps> = ({
  selectedEntityId,
  selectedEntityType,
  playerFactionId,
  entities,
  onUnitSpawn
}) => {
  if (!selectedEntityId || !selectedEntityType) {
    return (
      <div className="selection-panel empty">
        <p>No entity selected</p>
      </div>
    );
  }

  const entity = entities[selectedEntityId];
  
  if (!entity) {
    return (
      <div className="selection-panel empty">
        <p>Entity not found</p>
      </div>
    );
  }

  // IMPORTANT: Verify that the entity belongs to the player
  // This prevents showing enemy building UIs
  if (entity.factionId !== playerFactionId) {
    const faction = getFaction(entity.factionId);
    return (
      <div className="selection-panel enemy">
        <h3>Enemy {faction?.name || 'Unit'}</h3>
        <div className="health-bar">
          <div 
            className="health-fill"
            style={{ width: `${(entity.health / entity.maxHealth) * 100}%` }}
          />
        </div>
        <p>Health: {entity.health}/{entity.maxHealth}</p>
      </div>
    );
  }

  // Show building UI if it's a building
  if (selectedEntityType === 'building') {
    return (
      <div className="selection-panel building">
        <Building
          buildingId={entity.id}
          buildingType={entity.type}
          playerFactionId={playerFactionId} // Pass player's faction ID
          onUnitSpawn={onUnitSpawn}
        />
        <div className="health-bar">
          <div 
            className="health-fill"
            style={{ width: `${(entity.health / entity.maxHealth) * 100}%` }}
          />
        </div>
        <p>Health: {entity.health}/{entity.maxHealth}</p>
      </div>
    );
  }

  // Show unit UI
  const faction = getFaction(playerFactionId);
  const unit = faction?.units[entity.type];
  const unitPreview = getUnitPreview(playerFactionId, entity.type);

  return (
    <div className="selection-panel unit">
      <h3>{unit?.name || entity.type}</h3>
      {unitPreview && (
        <img 
          src={unitPreview} 
          alt={unit?.name || entity.type}
          className="unit-portrait"
        />
      )}
      <div className="health-bar">
        <div 
          className="health-fill"
          style={{ width: `${(entity.health / entity.maxHealth) * 100}%` }}
        />
      </div>
      <p>Health: {entity.health}/{entity.maxHealth}</p>
    </div>
  );
};

export default SelectionPanel;
