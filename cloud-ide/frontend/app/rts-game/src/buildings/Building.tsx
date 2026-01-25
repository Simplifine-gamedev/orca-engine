'use client';

import React, { useState } from 'react';
import { Building as BuildingType } from '../game/types';

interface BuildingProps {
  building: BuildingType;
  index: number;
}

export function Building({ building, index }: BuildingProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div 
      className="relative p-3 bg-gray-800 border-2 border-gray-600 rounded-lg hover:border-blue-500 transition-all cursor-pointer"
      onClick={() => setShowDetails(!showDetails)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{building.icon}</span>
          <div>
            <h4 className="text-sm font-bold text-white">{building.name}</h4>
            <span className="text-xs text-gray-400">#{index + 1}</span>
          </div>
        </div>
        
        {/* Status indicator */}
        <div className="flex flex-col items-end">
          <span className="text-xs text-green-400 font-semibold">● ACTIVE</span>
          {building.produces && (
            <div className="text-xs text-gray-400 mt-1">
              Producing
            </div>
          )}
        </div>
      </div>

      {/* Production Display */}
      {building.produces && (
        <div className="mt-3 p-2 bg-green-900/20 rounded border border-green-700">
          <div className="text-xs font-semibold text-green-400 mb-1">PRODUCING:</div>
          <div className="flex gap-3">
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
                  <span className="text-green-400 font-semibold text-xs">+{income}/sec</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Expanded Details */}
      {showDetails && (
        <div className="mt-3 p-3 bg-gray-900 rounded border border-gray-700">
          <h5 className="text-xs font-bold text-white mb-2">BUILDING DETAILS</h5>
          
          <div className="text-xs text-gray-300 mb-3">
            {building.description}
          </div>

          <div className="mb-2">
            <div className="text-xs font-semibold text-gray-400 mb-1">ORIGINAL COST:</div>
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
                    <span className="text-gray-400">{cost}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {building.produces && (
            <div className="mb-2">
              <div className="text-xs font-semibold text-gray-400 mb-1">RESOURCE PRODUCTION:</div>
              <div className="text-xs text-gray-300">
                {Object.entries(building.produces).map(([resource, income]) => (
                  <div key={resource} className="mb-1">
                    • {income} {resource} per second
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-3 p-2 bg-blue-900/20 rounded border border-blue-700">
            <div className="text-xs font-semibold text-blue-400 mb-1">💡 TIP:</div>
            <div className="text-xs text-gray-300">
              {building.type === 'resource' && 
                'Build more of these to increase your resource income!'
              }
              {building.type === 'military' && 
                'This building allows you to train military units.'
              }
              {building.type === 'main' && 
                'Main building that unlocks advanced features.'
              }
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface BuildingListProps {
  buildings: BuildingType[];
}

export function BuildingList({ buildings }: BuildingListProps) {
  if (buildings.length === 0) {
    return (
      <div className="p-8 bg-gray-800 rounded-lg border-2 border-dashed border-gray-600 text-center">
        <span className="text-6xl mb-4 block">🏗️</span>
        <h3 className="text-xl font-bold text-white mb-2">No Buildings Yet</h3>
        <p className="text-gray-400">
          Start building structures to generate resources and train units!
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {buildings.map((building, index) => (
        <Building key={`${building.id}-${index}`} building={building} index={index} />
      ))}
    </div>
  );
}
