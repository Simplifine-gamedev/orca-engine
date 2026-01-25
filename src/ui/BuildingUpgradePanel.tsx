/**
 * Building Upgrade Panel UI Component
 * Displays building upgrades and allows players to upgrade their structures
 */

import React, { useState, useEffect } from 'react';
import { 
  gameStore, 
  Building, 
  BuildingType,
  BUILDING_UPGRADES,
  BuildingUpgrade 
} from '../store/gameStore';

interface BuildingUpgradePanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const BuildingUpgradePanel: React.FC<BuildingUpgradePanelProps> = ({ isOpen, onClose }) => {
  const [gameState, setGameState] = useState(gameStore.getState());
  const [selectedBuilding, setSelectedBuilding] = useState<Building | null>(null);

  useEffect(() => {
    const unsubscribe = gameStore.subscribe(setGameState);
    return unsubscribe;
  }, []);

  if (!isOpen) return null;

  const getNextUpgrade = (building: Building): BuildingUpgrade | null => {
    return BUILDING_UPGRADES.find(
      u => u.buildingType === building.type && u.level === building.level + 1
    ) || null;
  };

  const handleUpgrade = (buildingId: string) => {
    if (gameStore.upgradeBuilding(buildingId)) {
      console.log('Building upgraded successfully');
    }
  };

  const getBuildingIcon = (type: BuildingType): string => {
    const icons: Record<BuildingType, string> = {
      [BuildingType.TOWN_CENTER]: '🏛️',
      [BuildingType.BARRACKS]: '⚔️',
      [BuildingType.ARCHERY_RANGE]: '🏹',
      [BuildingType.STABLE]: '🐴',
      [BuildingType.WORKSHOP]: '🔨',
      [BuildingType.TEMPLE]: '⛪',
      [BuildingType.MARKET]: '🏪',
      [BuildingType.BLACKSMITH]: '⚒️',
      [BuildingType.ACADEMY]: '📚',
      [BuildingType.DEFENSE_TOWER]: '🗼',
    };
    return icons[type] || '🏢';
  };

  const formatBuildingName = (type: BuildingType): string => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const canAffordUpgrade = (upgrade: BuildingUpgrade): boolean => {
    return gameStore.canAfford(upgrade.cost);
  };

  const isUnlocked = (buildingType: BuildingType): boolean => {
    return gameStore.canBuildBuilding(buildingType);
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border-2 border-gray-700 rounded-lg w-4/5 h-5/6 flex flex-col shadow-2xl">
        {/* Header */}
        <div className="bg-gray-800 border-b-2 border-gray-700 p-4 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">Building Upgrades</h2>
            <p className="text-sm text-gray-400">Upgrade your structures to unlock new capabilities</p>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-white font-semibold"
          >
            Close
          </button>
        </div>

        {/* Resource Display */}
        <div className="bg-gray-800 border-b border-gray-700 p-3 flex gap-6">
          <div className="flex items-center gap-2">
            <span className="text-yellow-500 font-bold">Gold:</span>
            <span className="text-white">{gameState.resources.gold}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-yellow-700 font-bold">Wood:</span>
            <span className="text-white">{gameState.resources.wood}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-400 font-bold">Stone:</span>
            <span className="text-white">{gameState.resources.stone}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-green-500 font-bold">Food:</span>
            <span className="text-white">{gameState.resources.food}</span>
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-blue-400 font-bold">Population:</span>
            <span className="text-white">{gameState.population.current} / {gameState.population.max}</span>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Buildings List */}
          <div className="w-1/2 overflow-y-auto p-4 border-r border-gray-700">
            <h3 className="text-xl font-bold text-white mb-4">Your Buildings</h3>
            <div className="space-y-3">
              {gameState.buildings.map(building => {
                const nextUpgrade = getNextUpgrade(building);
                const maxLevel = !nextUpgrade;
                const affordable = nextUpgrade ? canAffordUpgrade(nextUpgrade) : false;

                return (
                  <div
                    key={building.id}
                    onClick={() => setSelectedBuilding(building)}
                    className={`p-4 bg-gray-800 border-2 rounded-lg cursor-pointer transition-all hover:scale-105 ${
                      selectedBuilding?.id === building.id
                        ? 'border-blue-500 ring-2 ring-blue-500'
                        : 'border-gray-700'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-4xl">{getBuildingIcon(building.type)}</span>
                        <div>
                          <h4 className="text-lg font-bold text-white">
                            {formatBuildingName(building.type)}
                          </h4>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-sm text-gray-400">Level {building.level}</span>
                            {!maxLevel && (
                              <span className="text-xs text-blue-400">→ Level {building.level + 1}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      {maxLevel ? (
                        <div className="bg-purple-600 text-white px-3 py-1 rounded text-xs font-bold">
                          MAX LEVEL
                        </div>
                      ) : (
                        <div className={`px-3 py-1 rounded text-xs font-bold ${
                          affordable ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
                        }`}>
                          {affordable ? 'CAN UPGRADE' : 'NEED RESOURCES'}
                        </div>
                      )}
                    </div>

                    {nextUpgrade && (
                      <div className="mt-3 pt-3 border-t border-gray-700">
                        <div className="text-xs text-gray-400 mb-2">
                          Upgrade Cost: {nextUpgrade.cost.gold}g, {nextUpgrade.cost.wood}w, {nextUpgrade.cost.stone}s
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleUpgrade(building.id);
                          }}
                          disabled={!affordable}
                          className={`w-full py-2 rounded font-semibold ${
                            affordable
                              ? 'bg-green-600 hover:bg-green-700 text-white'
                              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                          }`}
                        >
                          {affordable ? 'Upgrade Now' : 'Cannot Afford'}
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Building Details */}
          <div className="w-1/2 overflow-y-auto p-4">
            {selectedBuilding ? (
              <>
                <div className="text-center mb-6">
                  <span className="text-6xl">{getBuildingIcon(selectedBuilding.type)}</span>
                  <h3 className="text-2xl font-bold text-white mt-2">
                    {formatBuildingName(selectedBuilding.type)}
                  </h3>
                  <p className="text-gray-400">Level {selectedBuilding.level}</p>
                </div>

                {(() => {
                  const nextUpgrade = getNextUpgrade(selectedBuilding);
                  if (!nextUpgrade) {
                    return (
                      <div className="bg-purple-900/30 border-2 border-purple-600 rounded-lg p-6 text-center">
                        <p className="text-2xl font-bold text-purple-400 mb-2">Maximum Level Reached!</p>
                        <p className="text-gray-300">This building is fully upgraded.</p>
                      </div>
                    );
                  }

                  const affordable = canAffordUpgrade(nextUpgrade);

                  return (
                    <>
                      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 mb-4">
                        <h4 className="text-lg font-bold text-white mb-3">Next Upgrade: Level {nextUpgrade.level}</h4>
                        
                        <div className="mb-4">
                          <h5 className="text-sm font-semibold text-gray-400 mb-2">Cost</h5>
                          <div className="grid grid-cols-2 gap-2">
                            {nextUpgrade.cost.gold > 0 && (
                              <div className="flex justify-between bg-gray-900 p-2 rounded">
                                <span className="text-yellow-500">Gold:</span>
                                <span className={gameState.resources.gold >= nextUpgrade.cost.gold ? 'text-white' : 'text-red-400'}>
                                  {nextUpgrade.cost.gold}
                                </span>
                              </div>
                            )}
                            {nextUpgrade.cost.wood > 0 && (
                              <div className="flex justify-between bg-gray-900 p-2 rounded">
                                <span className="text-yellow-700">Wood:</span>
                                <span className={gameState.resources.wood >= nextUpgrade.cost.wood ? 'text-white' : 'text-red-400'}>
                                  {nextUpgrade.cost.wood}
                                </span>
                              </div>
                            )}
                            {nextUpgrade.cost.stone > 0 && (
                              <div className="flex justify-between bg-gray-900 p-2 rounded">
                                <span className="text-gray-400">Stone:</span>
                                <span className={gameState.resources.stone >= nextUpgrade.cost.stone ? 'text-white' : 'text-red-400'}>
                                  {nextUpgrade.cost.stone}
                                </span>
                              </div>
                            )}
                            {nextUpgrade.cost.food > 0 && (
                              <div className="flex justify-between bg-gray-900 p-2 rounded">
                                <span className="text-green-500">Food:</span>
                                <span className={gameState.resources.food >= nextUpgrade.cost.food ? 'text-white' : 'text-red-400'}>
                                  {nextUpgrade.cost.food}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="mb-4">
                          <h5 className="text-sm font-semibold text-gray-400 mb-2">Benefits</h5>
                          <ul className="space-y-2">
                            {nextUpgrade.benefits.map((benefit, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-green-400 mt-1">✓</span>
                                <span className="text-gray-300">{benefit}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {nextUpgrade.unlocks && nextUpgrade.unlocks.length > 0 && (
                          <div className="mb-4">
                            <h5 className="text-sm font-semibold text-gray-400 mb-2">Unlocks</h5>
                            <div className="flex flex-wrap gap-2">
                              {nextUpgrade.unlocks.map(unlock => (
                                <div key={unlock} className="bg-blue-900/30 border border-blue-600 px-3 py-1 rounded">
                                  <span className="text-sm">{getBuildingIcon(unlock)}</span>
                                  <span className="text-xs text-blue-400 ml-1">
                                    {formatBuildingName(unlock)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        <button
                          onClick={() => handleUpgrade(selectedBuilding.id)}
                          disabled={!affordable}
                          className={`w-full py-3 rounded font-bold text-lg ${
                            affordable
                              ? 'bg-green-600 hover:bg-green-700 text-white'
                              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                          }`}
                        >
                          {affordable ? 'Upgrade Building' : 'Insufficient Resources'}
                        </button>
                      </div>

                      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                        <h5 className="text-sm font-semibold text-gray-400 mb-2">All Available Upgrades</h5>
                        <div className="space-y-2 max-h-60 overflow-y-auto">
                          {BUILDING_UPGRADES
                            .filter(u => u.buildingType === selectedBuilding.type)
                            .map(upgrade => {
                              const isCompleted = selectedBuilding.level >= upgrade.level;
                              const isCurrent = upgrade.level === selectedBuilding.level + 1;
                              
                              return (
                                <div
                                  key={`${upgrade.buildingType}-${upgrade.level}`}
                                  className={`p-2 rounded ${
                                    isCurrent
                                      ? 'bg-blue-900/30 border border-blue-600'
                                      : isCompleted
                                      ? 'bg-green-900/30 border border-green-600'
                                      : 'bg-gray-900 border border-gray-700'
                                  }`}
                                >
                                  <div className="flex items-center justify-between">
                                    <span className="text-white font-semibold">Level {upgrade.level}</span>
                                    {isCompleted && <span className="text-green-400 text-xs">✓ Completed</span>}
                                    {isCurrent && <span className="text-blue-400 text-xs">→ Next</span>}
                                  </div>
                                  <div className="text-xs text-gray-400 mt-1">
                                    {upgrade.benefits.join(', ')}
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    </>
                  );
                })()}
              </>
            ) : (
              <div className="text-center text-gray-400 mt-10">
                <p className="text-xl mb-2">Select a building to view upgrade options</p>
                <p className="text-sm">Click on any building in the list to see details</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer Stats */}
        <div className="bg-gray-800 border-t-2 border-gray-700 p-3 flex gap-6 text-sm">
          <div>
            <span className="text-gray-400">Total Buildings:</span>
            <span className="text-white ml-2 font-semibold">{gameState.buildings.length}</span>
          </div>
          <div>
            <span className="text-gray-400">Upgradeable:</span>
            <span className="text-green-400 ml-2 font-semibold">
              {gameState.buildings.filter(b => getNextUpgrade(b) !== null).length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Max Level:</span>
            <span className="text-purple-400 ml-2 font-semibold">
              {gameState.buildings.filter(b => getNextUpgrade(b) === null).length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BuildingUpgradePanel;
