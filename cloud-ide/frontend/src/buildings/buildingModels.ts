// 3D Model management for buildings
import { BuildingType } from '../types/game';
import { BUILDING_TYPES } from './buildingTypes';

export interface BuildingModel {
  id: string;
  modelPath: string;
  thumbnailPath: string;
  scale: number;
  rotationOffset: number;
  heightOffset: number;
}

// 3D model configurations for each building type
export const BUILDING_MODELS: Record<string, BuildingModel> = {
  blacksmith: {
    id: 'blacksmith',
    modelPath: '/models/buildings/blacksmith.glb',
    thumbnailPath: '/images/buildings/blacksmith_preview.png',
    scale: 1.0,
    rotationOffset: 0,
    heightOffset: 0,
  },
  
  town_hall: {
    id: 'town_hall',
    modelPath: '/models/buildings/town_hall.glb',
    thumbnailPath: '/images/buildings/town_hall_preview.png',
    scale: 1.5,
    rotationOffset: 0,
    heightOffset: 0,
  },
  
  barracks: {
    id: 'barracks',
    modelPath: '/models/buildings/barracks.glb',
    thumbnailPath: '/images/buildings/barracks_preview.png',
    scale: 1.0,
    rotationOffset: 0,
    heightOffset: 0,
  },
  
  archery_range: {
    id: 'archery_range',
    modelPath: '/models/buildings/archery_range.glb',
    thumbnailPath: '/images/buildings/archery_range_preview.png',
    scale: 1.0,
    rotationOffset: 0,
    heightOffset: 0,
  },
  
  farm: {
    id: 'farm',
    modelPath: '/models/buildings/farm.glb',
    thumbnailPath: '/images/buildings/farm_preview.png',
    scale: 0.8,
    rotationOffset: 0,
    heightOffset: 0,
  },
};

/**
 * Get the 3D model configuration for a building
 */
export const getBuildingModel = (buildingId: string): BuildingModel | null => {
  return BUILDING_MODELS[buildingId] || null;
};

/**
 * Get the thumbnail path for a building
 */
export const getBuildingThumbnail = (buildingId: string): string => {
  const model = BUILDING_MODELS[buildingId];
  if (model) {
    return model.thumbnailPath;
  }
  // Fallback to placeholder
  return '/images/buildings/placeholder.png';
};

/**
 * Preload a building model (for performance)
 */
export const preloadBuildingModel = async (buildingId: string): Promise<void> => {
  const model = BUILDING_MODELS[buildingId];
  if (!model) return;
  
  // In a real implementation, this would load the 3D model
  // For now, just preload the thumbnail
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = () => resolve(); // Fail gracefully
    img.src = model.thumbnailPath;
  });
};

/**
 * Generate fallback preview for buildings without 3D models
 */
export const generateBuildingPreview = (buildingType: BuildingType): string => {
  // Return a data URL for a simple colored square based on building category
  const colors: Record<string, string> = {
    military: '#DC2626',
    economic: '#16A34A',
    research: '#2563EB',
    defense: '#9333EA',
  };
  
  const color = colors[buildingType.category] || '#6B7280';
  
  // Simple SVG fallback
  return `data:image/svg+xml,${encodeURIComponent(`
    <svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">
      <rect width="128" height="128" fill="${color}"/>
      <text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="white" font-size="14" font-family="Arial">
        ${buildingType.name}
      </text>
    </svg>
  `)}`;
};
