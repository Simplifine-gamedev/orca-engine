import React, { useState, useEffect } from 'react';
import { Building, BuildingType } from '../types/buildings';
import { GameStore } from '../store/gameStore';

interface BuildingPanelProps {
  gameStore: GameStore;
  onClose?: () => void;
}

export const BuildingPanel: React.FC<BuildingPanelProps> = ({
  gameStore,
  onClose,
}) => {
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [unlockedBuildings, setUnlockedBuildings] = useState<BuildingType[]>([]);
  const [resources, setResources] = useState(gameStore.getResources());

  useEffect(() => {
    updateBuildings();
    const interval = setInterval(() => {
      updateBuildings();
      setResources(gameStore.getResources());
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const updateBuildings = () => {
    const allBuildings = gameStore.getAllBuildingDefinitions();
    setBuildings(allBuildings);
    setUnlockedBuildings(gameStore.getUnlockedBuildings());
  };

  const formatCost = (cost: any): string => {
    const parts: string[] = [];
    if (cost.gold) parts.push(`💰${cost.gold}`);
    if (cost.wood) parts.push(`🪵${cost.wood}`);
    if (cost.stone) parts.push(`🪨${cost.stone}`);
    if (cost.food) parts.push(`🌾${cost.food}`);
    return parts.join(' ');
  };

  return (
    <div className="building-panel bg-gray-900 text-white p-6 rounded-lg shadow-2xl max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Buildings & Upgrades</h1>
        {onClose && (
          <button
            onClick={onClose}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          >
            Close
          </button>
        )}
      </div>

      {/* Resources Display */}
      <div className="bg-gray-800 p-4 rounded mb-6">
        <div className="flex justify-around text-sm">
          <div className="flex items-center">
            <span className="text-yellow-400 mr-2">💰</span>
            <span>Gold: {Math.floor(resources.gold)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-brown-400 mr-2">🪵</span>
            <span>Wood: {Math.floor(resources.wood)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-gray-400 mr-2">🪨</span>
            <span>Stone: {Math.floor(resources.stone)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-green-400 mr-2">🌾</span>
            <span>Food: {Math.floor(resources.food)}</span>
          </div>
        </div>
      </div>

      {/* Building List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
        {buildings.map((building) => {
          const isUnlocked = unlockedBuildings.includes(building.type);
          const canUpgrade = gameStore.canUpgradeBuilding(building.type);
          const canAffordBase = gameStore.canAffordResources(building.baseCost);
          const canAffordUpgrade = gameStore.canAffordResources(
            building.upgradeCost
          );

          return (
            <div
              key={building.id}
              className={`bg-gray-800 p-4 rounded border-2 ${
                isUnlocked ? 'border-green-500' : 'border-gray-600'
              }`}
            >
              {/* Building Header */}
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-lg font-semibold">{building.name}</h3>
                <span
                  className={`px-2 py-1 rounded text-xs ${
                    isUnlocked ? 'bg-green-500' : 'bg-gray-500'
                  }`}
                >
                  {isUnlocked ? 'Unlocked' : 'Locked'}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-300 mb-3">{building.description}</p>

              {/* Level Info */}
              <div className="mb-3">
                <div className="text-sm">
                  <span className="text-gray-400">Max Level:</span>{' '}
                  <span className="text-yellow-400">{building.maxLevel}</span>
                </div>
              </div>

              {/* Effects */}
              <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Effects per Level:
                </h4>
                {building.effects.map((effect, idx) => (
                  <div key={idx} className="text-xs text-green-400">
                    • {effect.description}
                  </div>
                ))}
              </div>

              {/* Base Cost */}
              <div className="mb-2">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Base Build Cost:
                </h4>
                <div className="text-xs">{formatCost(building.baseCost)}</div>
                <div className="text-xs text-gray-400">
                  ⏱️ {building.buildTime}s
                </div>
              </div>

              {/* Upgrade Cost */}
              <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Upgrade Cost (per level):
                </h4>
                <div className="text-xs">{formatCost(building.upgradeCost)}</div>
                <div className="text-xs text-gray-400">
                  ⏱️ {building.upgradeTime}s
                </div>
              </div>

              {/* Prerequisites */}
              {building.unlocksAt.length > 0 && !isUnlocked && (
                <div className="mb-3">
                  <h4 className="text-xs font-semibold text-gray-400 mb-1">
                    Unlock Requirements:
                  </h4>
                  <div className="text-xs text-yellow-400">
                    {building.unlocksAt.map((prereq, idx) => (
                      <div key={idx}>
                        • {prereq.buildingType && `${prereq.buildingType} Level ${prereq.level}`}
                        {prereq.researchId && `Research: ${prereq.researchId}`}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Status Message */}
              <div className="w-full py-2 bg-blue-600 rounded text-center text-sm font-semibold">
                {isUnlocked ? `Available (Max Level ${building.maxLevel})` : 'Complete requirements to unlock'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default BuildingPanel;
