/**
 * Unit Upgrade Panel UI Component
 * Displays unit upgrades (armor, weapons, etc.) and hero unit creation
 */

import React, { useState, useEffect } from 'react';
import { 
  gameStore, 
  UnitType,
  UNIT_UPGRADES,
  UnitUpgrade 
} from '../store/gameStore';

interface UnitUpgradePanelProps {
  isOpen: boolean;
  onClose: () => void;
}

type UpgradeCategory = 'infantry' | 'ranged' | 'cavalry' | 'special' | 'heroes';

export const UnitUpgradePanel: React.FC<UnitUpgradePanelProps> = ({ isOpen, onClose }) => {
  const [gameState, setGameState] = useState(gameStore.getState());
  const [selectedCategory, setSelectedCategory] = useState<UpgradeCategory>('infantry');
  const [selectedUpgrade, setSelectedUpgrade] = useState<UnitUpgrade | null>(null);

  useEffect(() => {
    const unsubscribe = gameStore.subscribe(setGameState);
    return unsubscribe;
  }, []);

  if (!isOpen) return null;

  const getUnitIcon = (unitType: UnitType): string => {
    const icons: Record<UnitType, string> = {
      [UnitType.WORKER]: '👷',
      [UnitType.SWORDSMAN]: '🗡️',
      [UnitType.ARCHER]: '🏹',
      [UnitType.CAVALRY]: '🐎',
      [UnitType.SIEGE_ENGINE]: '🎯',
      [UnitType.MAGE]: '🧙',
      [UnitType.HERO_WARRIOR]: '⚔️',
      [UnitType.HERO_ARCHER]: '🏹',
      [UnitType.HERO_MAGE]: '🔮',
    };
    return icons[unitType] || '👤';
  };

  const formatUnitName = (unitType: UnitType): string => {
    return unitType.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const getUpgradesByCategory = (category: UpgradeCategory): UnitUpgrade[] => {
    const categoryMap: Record<UpgradeCategory, UnitType[]> = {
      infantry: [UnitType.SWORDSMAN],
      ranged: [UnitType.ARCHER],
      cavalry: [UnitType.CAVALRY],
      special: [UnitType.MAGE, UnitType.SIEGE_ENGINE],
      heroes: [UnitType.HERO_WARRIOR, UnitType.HERO_ARCHER, UnitType.HERO_MAGE],
    };

    const units = categoryMap[category];
    return UNIT_UPGRADES.filter(u => units.includes(u.unitType));
  };

  const hasUpgrade = (upgrade: UnitUpgrade): boolean => {
    return gameStore.hasUnitUpgrade(upgrade.unitType, upgrade.upgradeType, upgrade.level);
  };

  const canAfford = (upgrade: UnitUpgrade): boolean => {
    return gameStore.canAfford(upgrade.cost);
  };

  const handleUpgrade = (upgrade: UnitUpgrade) => {
    if (gameStore.upgradeUnit(upgrade.unitType, upgrade.upgradeType, upgrade.level)) {
      console.log('Unit upgraded successfully');
    }
  };

  const handleCreateHero = (heroType: UnitType) => {
    if (gameStore.createHeroUnit(heroType)) {
      console.log('Hero created successfully');
    }
  };

  const getTownCenterLevel = (): number => {
    const townCenter = gameState.buildings.find(b => b.type === 'town_center');
    return townCenter?.level || 1;
  };

  const canCreateHeroes = (): boolean => {
    return getTownCenterLevel() >= 5;
  };

  const getUpgradeTypeColor = (type: UnitUpgrade['upgradeType']): string => {
    const colors = {
      armor: 'text-gray-400',
      weapon: 'text-red-400',
      health: 'text-green-400',
      speed: 'text-blue-400',
      special: 'text-purple-400',
    };
    return colors[type] || 'text-gray-400';
  };

  const getUpgradeTypeIcon = (type: UnitUpgrade['upgradeType']): string => {
    const icons = {
      armor: '🛡️',
      weapon: '⚔️',
      health: '❤️',
      speed: '⚡',
      special: '✨',
    };
    return icons[type] || '📊';
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border-2 border-gray-700 rounded-lg w-5/6 h-5/6 flex flex-col shadow-2xl">
        {/* Header */}
        <div className="bg-gray-800 border-b-2 border-gray-700 p-4 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">Unit Upgrades & Heroes</h2>
            <p className="text-sm text-gray-400">Enhance your army with powerful upgrades</p>
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
        </div>

        {/* Category Tabs */}
        <div className="bg-gray-800 border-b border-gray-700 p-3 flex gap-2">
          <button
            onClick={() => setSelectedCategory('infantry')}
            className={`px-4 py-2 rounded font-semibold ${
              selectedCategory === 'infantry'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
            }`}
          >
            🗡️ Infantry
          </button>
          <button
            onClick={() => setSelectedCategory('ranged')}
            className={`px-4 py-2 rounded font-semibold ${
              selectedCategory === 'ranged'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
            }`}
          >
            🏹 Ranged
          </button>
          <button
            onClick={() => setSelectedCategory('cavalry')}
            className={`px-4 py-2 rounded font-semibold ${
              selectedCategory === 'cavalry'
                ? 'bg-yellow-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
            }`}
          >
            🐎 Cavalry
          </button>
          <button
            onClick={() => setSelectedCategory('special')}
            className={`px-4 py-2 rounded font-semibold ${
              selectedCategory === 'special'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
            }`}
          >
            🧙 Special Units
          </button>
          <button
            onClick={() => setSelectedCategory('heroes')}
            className={`px-4 py-2 rounded font-semibold ${
              selectedCategory === 'heroes'
                ? 'bg-orange-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
            }`}
          >
            ⚔️ Heroes
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden">
          {selectedCategory === 'heroes' ? (
            /* Hero Creation Panel */
            <div className="flex-1 overflow-y-auto p-6">
              {!canCreateHeroes() ? (
                <div className="bg-red-900/30 border-2 border-red-600 rounded-lg p-6 text-center">
                  <h3 className="text-2xl font-bold text-red-400 mb-3">Heroes Locked</h3>
                  <p className="text-gray-300 mb-4">
                    Upgrade your Town Center to Level 5 to unlock Hero units
                  </p>
                  <div className="text-white">
                    Current Town Center Level: <span className="font-bold text-yellow-400">{getTownCenterLevel()}</span>
                  </div>
                </div>
              ) : (
                <>
                  <h3 className="text-2xl font-bold text-white mb-4">Create Hero Units</h3>
                  <p className="text-gray-400 mb-6">
                    Heroes are powerful unique units that can turn the tide of battle. Each hero costs 2000 gold and 500 food.
                  </p>

                  <div className="grid grid-cols-3 gap-6">
                    {[
                      { type: UnitType.HERO_WARRIOR, name: 'Hero Warrior', desc: 'A legendary warrior with incredible melee combat skills', stats: { hp: 500, attack: 50, defense: 30 } },
                      { type: UnitType.HERO_ARCHER, name: 'Hero Archer', desc: 'Master of ranged combat with devastating accuracy', stats: { hp: 400, attack: 60, defense: 20 } },
                      { type: UnitType.HERO_MAGE, name: 'Hero Mage', desc: 'Archmage wielding powerful magical abilities', stats: { hp: 350, attack: 70, defense: 15 } },
                    ].map(hero => {
                      const cost = { gold: 2000, wood: 0, stone: 0, food: 500 };
                      const affordable = gameStore.canAfford(cost);
                      const heroCount = gameState.units.filter(u => u.type === hero.type).length;

                      return (
                        <div key={hero.type} className="bg-gray-800 border-2 border-orange-600 rounded-lg p-4">
                          <div className="text-center mb-4">
                            <span className="text-6xl">{getUnitIcon(hero.type)}</span>
                            <h4 className="text-xl font-bold text-white mt-2">{hero.name}</h4>
                            <p className="text-sm text-gray-400 mt-2">{hero.desc}</p>
                          </div>

                          <div className="bg-gray-900 rounded p-3 mb-4">
                            <h5 className="text-sm font-semibold text-gray-400 mb-2">Base Stats</h5>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span className="text-green-400">HP:</span>
                                <span className="text-white">{hero.stats.hp}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-red-400">Attack:</span>
                                <span className="text-white">{hero.stats.attack}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Defense:</span>
                                <span className="text-white">{hero.stats.defense}</span>
                              </div>
                            </div>
                          </div>

                          <div className="mb-4">
                            <h5 className="text-sm font-semibold text-gray-400 mb-2">Cost</h5>
                            <div className="flex justify-between text-sm">
                              <span className="text-yellow-500">2000 Gold</span>
                              <span className="text-green-500">500 Food</span>
                            </div>
                          </div>

                          {heroCount > 0 && (
                            <div className="bg-blue-900/30 border border-blue-600 rounded p-2 mb-3 text-center">
                              <span className="text-blue-400 text-sm font-semibold">
                                Active: {heroCount} {hero.name}(s)
                              </span>
                            </div>
                          )}

                          <button
                            onClick={() => handleCreateHero(hero.type)}
                            disabled={!affordable}
                            className={`w-full py-3 rounded font-bold ${
                              affordable
                                ? 'bg-orange-600 hover:bg-orange-700 text-white'
                                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                            }`}
                          >
                            {affordable ? 'Create Hero' : 'Insufficient Resources'}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          ) : (
            <>
              {/* Unit Upgrades List */}
              <div className="w-2/3 overflow-y-auto p-4">
                <div className="space-y-4">
                  {getUpgradesByCategory(selectedCategory).length === 0 ? (
                    <div className="text-center text-gray-400 mt-10">
                      <p>No upgrades available for this category</p>
                    </div>
                  ) : (
                    <>
                      {/* Group by unit type */}
                      {Array.from(new Set(getUpgradesByCategory(selectedCategory).map(u => u.unitType))).map(unitType => {
                        const unitUpgrades = getUpgradesByCategory(selectedCategory).filter(u => u.unitType === unitType);
                        
                        return (
                          <div key={unitType} className="bg-gray-800 border-2 border-gray-700 rounded-lg p-4">
                            <div className="flex items-center gap-3 mb-4">
                              <span className="text-4xl">{getUnitIcon(unitType)}</span>
                              <h3 className="text-xl font-bold text-white">{formatUnitName(unitType)}</h3>
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                              {unitUpgrades.map(upgrade => {
                                const purchased = hasUpgrade(upgrade);
                                const affordable = canAfford(upgrade);
                                const canPurchase = !purchased && affordable;

                                return (
                                  <div
                                    key={`${upgrade.unitType}-${upgrade.upgradeType}-${upgrade.level}`}
                                    onClick={() => setSelectedUpgrade(upgrade)}
                                    className={`p-3 border-2 rounded-lg cursor-pointer transition-all ${
                                      selectedUpgrade === upgrade
                                        ? 'ring-2 ring-blue-500'
                                        : ''
                                    } ${
                                      purchased
                                        ? 'bg-green-900/30 border-green-600'
                                        : 'bg-gray-900 border-gray-700 hover:border-gray-600'
                                    }`}
                                  >
                                    <div className="flex items-center justify-between mb-2">
                                      <div className="flex items-center gap-2">
                                        <span className="text-2xl">{getUpgradeTypeIcon(upgrade.upgradeType)}</span>
                                        <div>
                                          <div className={`text-sm font-semibold capitalize ${getUpgradeTypeColor(upgrade.upgradeType)}`}>
                                            {upgrade.upgradeType}
                                          </div>
                                          <div className="text-xs text-gray-400">Level {upgrade.level}</div>
                                        </div>
                                      </div>
                                      {purchased && (
                                        <div className="bg-green-600 text-white px-2 py-1 rounded text-xs font-bold">
                                          ✓
                                        </div>
                                      )}
                                    </div>

                                    <div className="text-xs text-gray-400 mb-2">
                                      {Object.entries(upgrade.effect).map(([stat, value]) => (
                                        <div key={stat}>
                                          {stat}: +{value}
                                        </div>
                                      ))}
                                    </div>

                                    <div className="text-xs text-gray-500 mb-2">
                                      Cost: {upgrade.cost.gold}g {upgrade.cost.wood}w {upgrade.cost.stone}s {upgrade.cost.food}f
                                    </div>

                                    {!purchased && (
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleUpgrade(upgrade);
                                        }}
                                        disabled={!canPurchase}
                                        className={`w-full py-2 rounded text-xs font-semibold ${
                                          canPurchase
                                            ? 'bg-green-600 hover:bg-green-700 text-white'
                                            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                        }`}
                                      >
                                        {affordable ? 'Purchase' : 'Cannot Afford'}
                                      </button>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </>
                  )}
                </div>
              </div>

              {/* Upgrade Details Panel */}
              <div className="w-1/3 bg-gray-800 border-l-2 border-gray-700 p-4 overflow-y-auto">
                {selectedUpgrade ? (
                  <>
                    <div className="text-center mb-6">
                      <span className="text-6xl">{getUpgradeTypeIcon(selectedUpgrade.upgradeType)}</span>
                      <h3 className={`text-2xl font-bold capitalize mt-2 ${getUpgradeTypeColor(selectedUpgrade.upgradeType)}`}>
                        {selectedUpgrade.upgradeType}
                      </h3>
                      <p className="text-gray-400">Level {selectedUpgrade.level}</p>
                    </div>

                    <div className="mb-6">
                      <h4 className="text-sm font-semibold text-gray-400 mb-2">Unit</h4>
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">{getUnitIcon(selectedUpgrade.unitType)}</span>
                        <span className="text-white font-semibold">{formatUnitName(selectedUpgrade.unitType)}</span>
                      </div>
                    </div>

                    <div className="mb-6">
                      <h4 className="text-sm font-semibold text-gray-400 mb-2">Cost</h4>
                      <div className="space-y-2">
                        {selectedUpgrade.cost.gold > 0 && (
                          <div className="flex justify-between bg-gray-900 p-2 rounded">
                            <span className="text-yellow-500">Gold:</span>
                            <span className={gameState.resources.gold >= selectedUpgrade.cost.gold ? 'text-white' : 'text-red-400'}>
                              {selectedUpgrade.cost.gold}
                            </span>
                          </div>
                        )}
                        {selectedUpgrade.cost.wood > 0 && (
                          <div className="flex justify-between bg-gray-900 p-2 rounded">
                            <span className="text-yellow-700">Wood:</span>
                            <span className={gameState.resources.wood >= selectedUpgrade.cost.wood ? 'text-white' : 'text-red-400'}>
                              {selectedUpgrade.cost.wood}
                            </span>
                          </div>
                        )}
                        {selectedUpgrade.cost.stone > 0 && (
                          <div className="flex justify-between bg-gray-900 p-2 rounded">
                            <span className="text-gray-400">Stone:</span>
                            <span className={gameState.resources.stone >= selectedUpgrade.cost.stone ? 'text-white' : 'text-red-400'}>
                              {selectedUpgrade.cost.stone}
                            </span>
                          </div>
                        )}
                        {selectedUpgrade.cost.food > 0 && (
                          <div className="flex justify-between bg-gray-900 p-2 rounded">
                            <span className="text-green-500">Food:</span>
                            <span className={gameState.resources.food >= selectedUpgrade.cost.food ? 'text-white' : 'text-red-400'}>
                              {selectedUpgrade.cost.food}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="mb-6">
                      <h4 className="text-sm font-semibold text-gray-400 mb-2">Effects</h4>
                      <div className="space-y-2">
                        {Object.entries(selectedUpgrade.effect).map(([stat, value]) => (
                          <div key={stat} className="flex justify-between bg-gray-900 p-2 rounded">
                            <span className="text-gray-300 capitalize">{stat}:</span>
                            <span className="text-green-400 font-semibold">+{value}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="pt-4 border-t border-gray-700">
                      <h4 className="text-sm font-semibold text-gray-400 mb-2">Status</h4>
                      <div className={`font-semibold ${
                        hasUpgrade(selectedUpgrade) ? 'text-green-400' : 'text-yellow-400'
                      }`}>
                        {hasUpgrade(selectedUpgrade) ? 'PURCHASED' : 'AVAILABLE'}
                      </div>
                    </div>

                    {!hasUpgrade(selectedUpgrade) && (
                      <button
                        onClick={() => handleUpgrade(selectedUpgrade)}
                        disabled={!canAfford(selectedUpgrade)}
                        className={`w-full py-3 rounded font-bold mt-6 ${
                          canAfford(selectedUpgrade)
                            ? 'bg-green-600 hover:bg-green-700 text-white'
                            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                      >
                        {canAfford(selectedUpgrade) ? 'Purchase Upgrade' : 'Insufficient Resources'}
                      </button>
                    )}
                  </>
                ) : (
                  <div className="text-center text-gray-400 mt-10">
                    <p>Select an upgrade to view details</p>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer Stats */}
        <div className="bg-gray-800 border-t-2 border-gray-700 p-3 flex gap-6 text-sm">
          <div>
            <span className="text-gray-400">Total Units:</span>
            <span className="text-white ml-2 font-semibold">{gameState.units.length}</span>
          </div>
          <div>
            <span className="text-gray-400">Heroes:</span>
            <span className="text-orange-400 ml-2 font-semibold">
              {gameState.units.filter(u => u.isHero).length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Purchased Upgrades:</span>
            <span className="text-green-400 ml-2 font-semibold">
              {gameState.unitUpgrades.size}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UnitUpgradePanel;
