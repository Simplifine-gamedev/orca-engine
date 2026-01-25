'use client'

import React from 'react';
import { Building as BuildingInstance } from '../types/game';
import { getBuildingModel, getBuildingThumbnail, generateBuildingPreview } from './buildingModels';
import { BUILDING_TYPES } from './buildingTypes';

interface BuildingProps {
  building: BuildingInstance;
  onClick?: () => void;
  isSelected?: boolean;
  showDetails?: boolean;
}

export const Building: React.FC<BuildingProps> = ({
  building,
  onClick,
  isSelected = false,
  showDetails = true,
}) => {
  const buildingType = building.type;
  const model = getBuildingModel(buildingType.id);
  const thumbnail = model ? model.thumbnailPath : generateBuildingPreview(buildingType);

  return (
    <div
      className={`
        relative rounded-lg overflow-hidden transition-all cursor-pointer
        ${isSelected ? 'ring-4 ring-blue-500 scale-105' : 'hover:scale-102'}
        ${building.health < buildingType.maxHealth * 0.3 ? 'ring-2 ring-red-500' : ''}
      `}
      onClick={onClick}
    >
      {/* Building Preview/Thumbnail */}
      <div className="relative w-full h-32 bg-gray-800 flex items-center justify-center overflow-hidden">
        <img
          src={thumbnail}
          alt={buildingType.name}
          className="w-full h-full object-cover"
          onError={(e) => {
            // Fallback to generated preview on error
            (e.target as HTMLImageElement).src = generateBuildingPreview(buildingType);
          }}
        />
        
        {/* Construction Progress Overlay */}
        {building.isConstructing && (
          <div className="absolute inset-0 bg-black bg-opacity-70 flex flex-col items-center justify-center">
            <div className="text-white text-sm mb-2">Constructing...</div>
            <div className="w-3/4 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all"
                style={{ width: `${building.constructionProgress}%` }}
              />
            </div>
            <div className="text-white text-xs mt-1">{Math.round(building.constructionProgress)}%</div>
          </div>
        )}
        
        {/* Health Bar */}
        {!building.isConstructing && building.health < buildingType.maxHealth && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-900">
            <div
              className={`h-full transition-all ${
                building.health / buildingType.maxHealth > 0.5
                  ? 'bg-green-500'
                  : building.health / buildingType.maxHealth > 0.25
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${(building.health / buildingType.maxHealth) * 100}%` }}
            />
          </div>
        )}
      </div>

      {/* Building Info */}
      {showDetails && (
        <div className="bg-gray-900 p-2">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-white font-semibold text-sm">{buildingType.name}</h3>
            <span className={`
              text-xs px-2 py-0.5 rounded-full
              ${buildingType.category === 'military' ? 'bg-red-900 text-red-200' : ''}
              ${buildingType.category === 'economic' ? 'bg-green-900 text-green-200' : ''}
              ${buildingType.category === 'research' ? 'bg-blue-900 text-blue-200' : ''}
              ${buildingType.category === 'defense' ? 'bg-purple-900 text-purple-200' : ''}
            `}>
              {buildingType.category}
            </span>
          </div>
          <p className="text-gray-400 text-xs line-clamp-2">{buildingType.description}</p>
          
          {/* Resource Cost Display */}
          <div className="flex gap-2 mt-2 text-xs">
            {buildingType.cost.gold && (
              <span className="text-yellow-400">
                💰 {buildingType.cost.gold}
              </span>
            )}
            {buildingType.cost.wood && (
              <span className="text-amber-600">
                🪵 {buildingType.cost.wood}
              </span>
            )}
            {buildingType.cost.stone && (
              <span className="text-gray-400">
                🪨 {buildingType.cost.stone}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

interface BuildingCardProps {
  buildingType: string;
  onBuild?: () => void;
  canAfford?: boolean;
}

export const BuildingCard: React.FC<BuildingCardProps> = ({
  buildingType,
  onBuild,
  canAfford = true,
}) => {
  const building = BUILDING_TYPES[buildingType];
  if (!building) return null;

  const thumbnail = getBuildingThumbnail(building.id);

  return (
    <div
      className={`
        relative rounded-lg overflow-hidden transition-all
        ${canAfford ? 'cursor-pointer hover:scale-105' : 'opacity-50 cursor-not-allowed'}
      `}
      onClick={() => canAfford && onBuild?.()}
    >
      <div className="relative w-full h-24 bg-gray-800 flex items-center justify-center">
        <img
          src={thumbnail}
          alt={building.name}
          className="w-full h-full object-cover"
          onError={(e) => {
            (e.target as HTMLImageElement).src = generateBuildingPreview(building);
          }}
        />
      </div>
      
      <div className="bg-gray-900 p-2">
        <h3 className="text-white font-semibold text-sm mb-1">{building.name}</h3>
        <div className="flex gap-2 text-xs">
          {building.cost.gold && (
            <span className={canAfford ? 'text-yellow-400' : 'text-red-400'}>
              💰 {building.cost.gold}
            </span>
          )}
          {building.cost.wood && (
            <span className={canAfford ? 'text-amber-600' : 'text-red-400'}>
              🪵 {building.cost.wood}
            </span>
          )}
          {building.cost.stone && (
            <span className={canAfford ? 'text-gray-400' : 'text-red-400'}>
              🪨 {building.cost.stone}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
