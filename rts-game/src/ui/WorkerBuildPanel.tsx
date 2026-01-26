import React, { useState } from 'react';
import { getFaction, getBuildingThumbnail } from '../config/factions';

export interface WorkerBuildPanelProps {
  workerId: string;
  playerFactionId: string;
  onBuildingPlace: (buildingType: string) => void;
}

export const WorkerBuildPanel: React.FC<WorkerBuildPanelProps> = ({
  workerId,
  playerFactionId,
  onBuildingPlace
}) => {
  const [selectedBuilding, setSelectedBuilding] = useState<string | null>(null);
  
  const faction = getFaction(playerFactionId);
  
  if (!faction) {
    console.error(`Invalid faction ID: ${playerFactionId}`);
    return <div>Error: Invalid faction</div>;
  }

  const handleBuildingSelect = (buildingId: string) => {
    setSelectedBuilding(buildingId);
    onBuildingPlace(buildingId);
  };

  return (
    <div className="worker-build-panel">
      <h3>Build Structure</h3>
      <div className="building-grid">
        {Object.entries(faction.buildings).map(([buildingId, building]) => {
          // Get faction-specific thumbnail
          const thumbnail = getBuildingThumbnail(playerFactionId, buildingId);

          return (
            <div
              key={buildingId}
              className={`building-button ${selectedBuilding === buildingId ? 'selected' : ''}`}
              onClick={() => handleBuildingSelect(buildingId)}
            >
              {thumbnail && (
                <img
                  src={thumbnail}
                  alt={building.name}
                  className="building-thumbnail"
                />
              )}
              <div className="building-info">
                <span className="building-name">{building.name}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default WorkerBuildPanel;
