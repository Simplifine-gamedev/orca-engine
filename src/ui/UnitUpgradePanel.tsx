import React, { useState, useEffect } from 'react';
import { Unit, UnitType, UnitUpgrade } from '../types/units';
import { GameStore } from '../store/gameStore';

interface UnitUpgradePanelProps {
  gameStore: GameStore;
  onClose?: () => void;
}

export const UnitUpgradePanel: React.FC<UnitUpgradePanelProps> = ({
  gameStore,
  onClose,
}) => {
  const [units, setUnits] = useState<Unit[]>([]);
  const [unlockedUnits, setUnlockedUnits] = useState<UnitType[]>([]);
  const [resources, setResources] = useState(gameStore.getResources());

  useEffect(() => {
    updateUnits();
    const interval = setInterval(() => {
      updateUnits();
      setResources(gameStore.getResources());
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const updateUnits = () => {
    const allUnits = gameStore.getAllUnitDefinitions();
    setUnits(allUnits);
    setUnlockedUnits(gameStore.getUnlockedUnits());
  };

  const formatCost = (cost: any): string => {
    const parts: string[] = [];
    if (cost.gold) parts.push(`💰${cost.gold}`);
    if (cost.wood) parts.push(`🪵${cost.wood}`);
    if (cost.stone) parts.push(`🪨${cost.stone}`);
    if (cost.food) parts.push(`🌾${cost.food}`);
    if (cost.mana) parts.push(`✨${cost.mana}`);
    return parts.join(' ');
  };

  const formatStats = (stats: any): JSX.Element => {
    return (
      <div className="grid grid-cols-2 gap-1 text-xs">
        <div>HP: {stats.hp}</div>
        <div>Attack: {stats.attackDamage}</div>
        <div>Armor: {stats.armor}</div>
        <div>Speed: {stats.movementSpeed}</div>
        <div>Attack Speed: {stats.attackSpeed}/s</div>
        <div>Range: {stats.attackRange}</div>
      </div>
    );
  };

  return (
    <div className="unit-upgrade-panel bg-gray-900 text-white p-6 rounded-lg shadow-2xl max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Unit Upgrades</h1>
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
          <div className="flex items-center">
            <span className="text-purple-400 mr-2">✨</span>
            <span>Mana: {Math.floor(resources.mana)}</span>
          </div>
        </div>
      </div>

      {/* Unit List */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
        {units.map((unit) => {
          const isUnlocked = unlockedUnits.includes(unit.type);
          const upgrades = gameStore.getUnitUpgrades(unit.type);
          const canAffordBase = gameStore.canAffordResources(unit.baseCost);

          return (
            <div
              key={unit.id}
              className={`bg-gray-800 p-4 rounded border-2 ${
                isUnlocked ? 'border-green-500' : 'border-gray-600'
              }`}
            >
              {/* Unit Header */}
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-lg font-semibold">{unit.name}</h3>
                <span
                  className={`px-2 py-1 rounded text-xs ${
                    isUnlocked ? 'bg-green-500' : 'bg-gray-500'
                  }`}
                >
                  {isUnlocked ? 'Unlocked' : 'Locked'}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-300 mb-3">{unit.description}</p>

              {/* Base Stats */}
              <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Base Stats:
                </h4>
                {formatStats(unit.baseStats)}
              </div>

              {/* Training Cost */}
              <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Training Cost:
                </h4>
                <div className="text-xs">{formatCost(unit.baseCost)}</div>
                <div className="text-xs text-gray-400">
                  ⏱️ {unit.trainingTime}s
                </div>
              </div>

              {/* Abilities */}
              {unit.abilities.length > 0 && (
                <div className="mb-3">
                  <h4 className="text-xs font-semibold text-gray-400 mb-1">
                    Abilities:
                  </h4>
                  {unit.abilities.map((ability, idx) => (
                    <div
                      key={idx}
                      className="text-xs text-purple-400 mb-1"
                    >
                      <div className="font-semibold">{ability.name}</div>
                      <div className="text-gray-400">
                        {ability.description} (CD: {ability.cooldown}s)
                        {ability.unlockResearchId && (
                          <span className="text-yellow-400">
                            {' '}
                            - Requires research
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Available Upgrades */}
              {isUnlocked && upgrades.length > 0 && (
                <div className="mb-3">
                  <h4 className="text-xs font-semibold text-gray-400 mb-2">
                    Available Upgrades ({upgrades.length}):
                  </h4>
                  {upgrades.map((upgrade) => (
                    <div
                      key={upgrade.id}
                      className="bg-gray-700 p-2 rounded mb-2"
                    >
                      <div className="text-sm font-semibold text-green-400">
                        Level {upgrade.level}: {upgrade.name}
                      </div>
                      <div className="text-xs text-gray-300 mb-1">
                        {upgrade.description}
                      </div>
                      <div className="text-xs mb-1">
                        Cost: {formatCost(upgrade.cost)}
                      </div>
                      <div className="text-xs text-gray-400 mb-1">
                        ⏱️ {upgrade.researchTime}s
                      </div>
                      <div className="text-xs text-blue-400">
                        Stat Bonuses:
                        {upgrade.statBonuses.hp && ` +${upgrade.statBonuses.hp} HP`}
                        {upgrade.statBonuses.attackDamage &&
                          ` +${upgrade.statBonuses.attackDamage} ATK`}
                        {upgrade.statBonuses.armor &&
                          ` +${upgrade.statBonuses.armor} ARM`}
                        {upgrade.statBonuses.movementSpeed &&
                          ` +${upgrade.statBonuses.movementSpeed} SPD`}
                        {upgrade.statBonuses.attackSpeed &&
                          ` +${upgrade.statBonuses.attackSpeed} AS`}
                        {upgrade.statBonuses.attackRange &&
                          ` +${upgrade.statBonuses.attackRange} RNG`}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Prerequisites */}
              {!isUnlocked && unit.prerequisites.length > 0 && (
                <div className="mb-3">
                  <h4 className="text-xs font-semibold text-gray-400 mb-1">
                    Unlock Requirements:
                  </h4>
                  <div className="text-xs text-yellow-400">
                    {unit.prerequisites.map((prereq, idx) => (
                      <div key={idx}>
                        {prereq.buildingType && (
                          <div>
                            • {prereq.buildingType} Level{' '}
                            {prereq.buildingLevel}
                          </div>
                        )}
                        {prereq.researchId && (
                          <div>• Research: {prereq.researchId}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Train Button */}
              {isUnlocked && (
                <button
                  onClick={() => {
                    if (gameStore.trainUnit(unit.type)) {
                      alert(`Training ${unit.name}!`);
                      updateUnits();
                    } else {
                      alert('Cannot train unit - check requirements!');
                    }
                  }}
                  disabled={!canAffordBase}
                  className={`w-full py-2 rounded font-semibold text-sm ${
                    canAffordBase
                      ? 'bg-green-600 hover:bg-green-700'
                      : 'bg-gray-600 cursor-not-allowed'
                  }`}
                >
                  {canAffordBase ? 'Train Unit' : 'Not Enough Resources'}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="mt-6 bg-gray-800 p-4 rounded">
        <h3 className="text-lg font-semibold mb-2">Unit Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Unlocked Units:</span>{' '}
            <span className="text-green-400">{unlockedUnits.length}</span>
          </div>
          <div>
            <span className="text-gray-400">Total Units:</span>{' '}
            <span className="text-blue-400">{units.length}</span>
          </div>
          <div>
            <span className="text-gray-400">Unlock Progress:</span>{' '}
            <span className="text-purple-400">
              {Math.floor((unlockedUnits.length / units.length) * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UnitUpgradePanel;
