'use client';

import React, { useState } from 'react';
import { useGame } from '../game/GameContext';
import { BUILDING_TYPES, ResourceCost, Building } from '../game/types';

interface BuildingCardProps {
  building: Building;
  canAfford: boolean;
  onBuild: () => void;
}

function BuildingCard({ building, canAfford, onBuild }: BuildingCardProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  const getCostColor = (resource: string, cost: number): string => {
    const { gameState } = useGame();
    const currentAmount = gameState.resources[resource as keyof typeof gameState.resources];
    return currentAmount >= cost ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div 
      className={`relative p-4 rounded-lg border-2 transition-all cursor-pointer ${
        canAfford 
          ? 'bg-gray-800 border-green-600 hover:bg-gray-700 hover:border-green-500' 
          : 'bg-gray-900 border-red-600 opacity-60 cursor-not-allowed'
      }`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      onClick={() => canAfford && onBuild()}
    >
      {/* Building Icon and Name */}
      <div className="flex items-center gap-3 mb-3">
        <span className="text-4xl">{building.icon}</span>
        <div>
          <h3 className="text-lg font-bold text-white">{building.name}</h3>
          <span className="text-xs text-gray-400">{building.type}</span>
        </div>
      </div>

      {/* Cost Display */}
      <div className="mb-2">
        <div className="text-xs font-semibold text-gray-400 mb-1">COST:</div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(building.cost).map(([resource, cost]) => {
            const icons: Record<string, string> = {
              gold: '💰',
              wood: '🪵',
              stone: '🪨',
              food: '🍖'
            };
            
            return (
              <div key={resource} className="flex items-center gap-1">
                <span>{icons[resource]}</span>
                <span className={`font-semibold ${getCostColor(resource, cost || 0)}`}>
                  {cost}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Production Info */}
      {building.produces && (
        <div className="mb-2 p-2 bg-green-900/30 rounded border border-green-700">
          <div className="text-xs font-semibold text-green-400 mb-1">PRODUCES:</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(building.produces).map(([resource, income]) => {
              const icons: Record<string, string> = {
                gold: '💰',
                wood: '🪵',
                stone: '🪨',
                food: '🍖'
              };
              
              return (
                <div key={resource} className="flex items-center gap-1">
                  <span>{icons[resource]}</span>
                  <span className="text-green-400 font-semibold">+{income}/sec</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Build Time */}
      <div className="text-xs text-gray-400">
        ⏱️ Build Time: {building.buildTime}s
      </div>

      {/* Affordability Indicator */}
      {!canAfford && (
        <div className="absolute top-2 right-2 px-2 py-1 bg-red-600 text-white text-xs font-bold rounded">
          CAN'T AFFORD
        </div>
      )}

      {canAfford && (
        <div className="absolute top-2 right-2 px-2 py-1 bg-green-600 text-white text-xs font-bold rounded">
          READY TO BUILD
        </div>
      )}

      {/* Detailed Tooltip */}
      {showTooltip && (
        <div className="absolute top-full left-0 mt-2 z-50 w-80 p-4 bg-gray-900 border-2 border-gray-700 rounded-lg shadow-xl">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-3xl">{building.icon}</span>
            <h4 className="font-bold text-white text-lg">{building.name}</h4>
          </div>
          <p className="text-sm text-gray-300 mb-3">{building.description}</p>
          
          {building.produces && (
            <div className="mb-3 p-2 bg-green-900/30 rounded">
              <div className="text-sm font-semibold text-green-400 mb-1">What it does:</div>
              <ul className="text-xs text-gray-300 list-disc list-inside">
                {Object.entries(building.produces).map(([resource, income]) => (
                  <li key={resource}>
                    Generates +{income} {resource} per second
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="text-xs text-gray-400">
            <div className="font-semibold mb-1">Usage Tips:</div>
            <ul className="list-disc list-inside">
              <li>Build early for resource advantage</li>
              <li>Place near resource deposits for efficiency</li>
              <li>Protect with defensive structures</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export function WorkerBuildPanel() {
  const { gameState, canAfford, buildBuilding } = useGame();

  const handleBuild = (buildingType: string) => {
    const success = buildBuilding(buildingType);
    if (success) {
      console.log(`Built ${buildingType}`);
    }
  };

  const resourceBuildings = Object.entries(BUILDING_TYPES).filter(
    ([_, building]) => building.type === 'resource'
  );

  const militaryBuildings = Object.entries(BUILDING_TYPES).filter(
    ([_, building]) => building.type === 'military' || building.type === 'main'
  );

  return (
    <div className="w-full p-6 bg-gray-800">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">Build Menu</h2>
          <p className="text-sm text-gray-400">
            💡 Click on buildings to construct them. Green border means you can afford it!
          </p>
        </div>

        {/* Resource Buildings */}
        <div className="mb-8">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>⛏️</span> Resource Buildings
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {resourceBuildings.map(([key, building]) => (
              <BuildingCard 
                key={key}
                building={building}
                canAfford={canAfford(building.cost)}
                onBuild={() => handleBuild(key)}
              />
            ))}
          </div>
        </div>

        {/* Military Buildings */}
        <div>
          <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span>⚔️</span> Military & Main Buildings
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {militaryBuildings.map(([key, building]) => (
              <BuildingCard 
                key={key}
                building={building}
                canAfford={canAfford(building.cost)}
                onBuild={() => handleBuild(key)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
