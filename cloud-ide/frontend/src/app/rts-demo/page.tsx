'use client'

import React, { useState, useEffect } from 'react';
import { Building, BuildingCard } from '../../buildings/Building';
import { ResearchPanel } from '../../ui/ResearchPanel';
import { BUILDING_TYPES } from '../../buildings/buildingTypes';
import { Building as BuildingInstance, GameResources } from '../../types/game';
import { researchStore } from '../../store/researchStore';

export default function RTSDemo() {
  const [selectedBuilding, setSelectedBuilding] = useState<string | null>('blacksmith');
  const [showResearchPanel, setShowResearchPanel] = useState(false);
  const [playerResources, setPlayerResources] = useState<GameResources>({
    gold: 500,
    wood: 400,
    stone: 300,
    food: 200,
  });

  // Simulate resource generation
  useEffect(() => {
    const interval = setInterval(() => {
      setPlayerResources(prev => ({
        gold: Math.min(999, prev.gold + 10),
        wood: Math.min(999, prev.wood + 8),
        stone: Math.min(999, prev.stone + 5),
        food: Math.min(999, prev.food + 3),
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Update research progress
  useEffect(() => {
    const interval = setInterval(() => {
      researchStore.updateResearch(1); // 1 second tick
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const mockBuildings: BuildingInstance[] = [
    {
      type: BUILDING_TYPES.BLACKSMITH,
      id: 'building_1',
      position: { x: 0, y: 0, z: 0 },
      health: BUILDING_TYPES.BLACKSMITH.maxHealth,
      isConstructing: false,
      constructionProgress: 100,
      ownerId: 'player1',
    },
    {
      type: BUILDING_TYPES.BARRACKS,
      id: 'building_2',
      position: { x: 10, y: 0, z: 0 },
      health: BUILDING_TYPES.BARRACKS.maxHealth * 0.7,
      isConstructing: false,
      constructionProgress: 100,
      ownerId: 'player1',
    },
    {
      type: BUILDING_TYPES.FARM,
      id: 'building_3',
      position: { x: 20, y: 0, z: 0 },
      health: BUILDING_TYPES.FARM.maxHealth,
      isConstructing: true,
      constructionProgress: 45,
      ownerId: 'player1',
    },
  ];

  const handleStartResearch = (techId: string) => {
    const tech = researchStore.getState().availableTechs.find(t => t.id === techId);
    if (!tech) return;

    // Deduct resources
    setPlayerResources(prev => ({
      gold: prev.gold - (tech.cost.gold || 0),
      wood: prev.wood - (tech.cost.wood || 0),
      stone: prev.stone - (tech.cost.stone || 0),
      food: prev.food - (tech.cost.food || 0),
    }));
  };

  const handleBuildBuilding = (buildingType: string) => {
    const building = BUILDING_TYPES[buildingType];
    if (!building) return;

    // Check if can afford
    const canAfford = 
      (!building.cost.gold || playerResources.gold >= building.cost.gold) &&
      (!building.cost.wood || playerResources.wood >= building.cost.wood) &&
      (!building.cost.stone || playerResources.stone >= building.cost.stone);

    if (canAfford) {
      setPlayerResources(prev => ({
        gold: prev.gold - (building.cost.gold || 0),
        wood: prev.wood - (building.cost.wood || 0),
        stone: prev.stone - (building.cost.stone || 0),
        food: prev.food,
      }));
      alert(`Building ${building.name}...`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Orca RTS Demo - Blacksmith Building</h1>
        <p className="text-gray-400">
          Demonstrating building system, 3D model previews, and research technology tree
        </p>
      </div>

      {/* Resources Bar */}
      <div className="bg-gray-900 rounded-lg p-4 mb-6 flex items-center justify-between">
        <div className="flex gap-6">
          <div className="flex items-center gap-2">
            <span className="text-2xl">💰</span>
            <div>
              <div className="text-xs text-gray-400">Gold</div>
              <div className="text-xl font-bold text-yellow-400">{playerResources.gold}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl">🪵</span>
            <div>
              <div className="text-xs text-gray-400">Wood</div>
              <div className="text-xl font-bold text-amber-600">{playerResources.wood}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl">🪨</span>
            <div>
              <div className="text-xs text-gray-400">Stone</div>
              <div className="text-xl font-bold text-gray-400">{playerResources.stone}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-2xl">🌾</span>
            <div>
              <div className="text-xs text-gray-400">Food</div>
              <div className="text-xl font-bold text-green-400">{playerResources.food}</div>
            </div>
          </div>
        </div>
        <div className="text-sm text-gray-400">
          Resources auto-generate every 2 seconds
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Buildings */}
        <div>
          <h2 className="text-xl font-bold mb-4">Your Buildings</h2>
          <div className="grid grid-cols-2 gap-4 mb-8">
            {mockBuildings.map(building => (
              <Building
                key={building.id}
                building={building}
                isSelected={selectedBuilding === building.id}
                onClick={() => {
                  setSelectedBuilding(building.id);
                  if (building.type.category === 'research') {
                    setShowResearchPanel(true);
                  }
                }}
              />
            ))}
          </div>

          <h2 className="text-xl font-bold mb-4">Build New Buildings</h2>
          <div className="grid grid-cols-2 gap-4">
            {Object.keys(BUILDING_TYPES).map(key => {
              const building = BUILDING_TYPES[key];
              const canAfford = 
                (!building.cost.gold || playerResources.gold >= building.cost.gold) &&
                (!building.cost.wood || playerResources.wood >= building.cost.wood) &&
                (!building.cost.stone || playerResources.stone >= building.cost.stone);
              
              return (
                <BuildingCard
                  key={building.id}
                  buildingType={building.id}
                  onBuild={() => handleBuildBuilding(building.id)}
                  canAfford={canAfford}
                />
              );
            })}
          </div>
        </div>

        {/* Right Column - Research Panel */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold">Research Technologies</h2>
            <button
              onClick={() => setShowResearchPanel(!showResearchPanel)}
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              {showResearchPanel ? 'Hide' : 'Show'} Panel
            </button>
          </div>
          
          {showResearchPanel ? (
            <ResearchPanel
              buildingId="blacksmith"
              buildingName="Blacksmith"
              playerResources={playerResources}
              onStartResearch={handleStartResearch}
            />
          ) : (
            <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-400">
              <div className="text-6xl mb-4">🔨</div>
              <p>Select a research building to view available technologies</p>
              <button
                onClick={() => setShowResearchPanel(true)}
                className="mt-4 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white"
              >
                Open Blacksmith Research
              </button>
            </div>
          )}

          {/* Info Panel */}
          <div className="mt-6 bg-gray-800 rounded-lg p-4">
            <h3 className="font-bold mb-2">Blacksmith Building Features:</h3>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>✅ 3D model preview system with fallback</li>
              <li>✅ Building thumbnail generation</li>
              <li>✅ Research technology tree with prerequisites</li>
              <li>✅ 7 blacksmith research technologies</li>
              <li>✅ Resource cost validation</li>
              <li>✅ Research progress tracking</li>
              <li>✅ Construction progress visualization</li>
              <li>✅ Health bar indicators</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Debug Controls */}
      <div className="mt-6 bg-gray-900 rounded-lg p-4">
        <h3 className="font-bold mb-2">Debug Controls</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setPlayerResources({ gold: 999, wood: 999, stone: 999, food: 999 })}
            className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm"
          >
            Max Resources
          </button>
          <button
            onClick={() => researchStore.reset()}
            className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
          >
            Reset Research
          </button>
          <button
            onClick={() => setPlayerResources({ gold: 50, wood: 50, stone: 50, food: 50 })}
            className="bg-yellow-600 hover:bg-yellow-700 px-3 py-1 rounded text-sm"
          >
            Low Resources
          </button>
        </div>
      </div>
    </div>
  );
}
