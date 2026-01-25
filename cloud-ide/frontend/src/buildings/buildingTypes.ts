// Building type definitions for Orca RTS
import { BuildingType } from '../types/game';

export const BUILDING_TYPES: Record<string, BuildingType> = {
  TOWN_HALL: {
    id: 'town_hall',
    name: 'Town Hall',
    description: 'Main building for resource gathering and unit production',
    cost: { gold: 500, wood: 500 },
    buildTime: 60,
    maxHealth: 2000,
    thumbnailPath: '/buildings/town_hall_thumb.png',
    category: 'economic',
  },
  
  BLACKSMITH: {
    id: 'blacksmith',
    name: 'Blacksmith',
    description: 'Research building for weapon and armor upgrades',
    cost: { gold: 150, wood: 100 },
    buildTime: 40,
    maxHealth: 800,
    modelPath: '/models/blacksmith.glb',
    thumbnailPath: '/buildings/blacksmith_thumb.png',
    category: 'research',
  },
  
  BARRACKS: {
    id: 'barracks',
    name: 'Barracks',
    description: 'Train infantry units',
    cost: { gold: 200, wood: 150 },
    buildTime: 50,
    maxHealth: 1200,
    thumbnailPath: '/buildings/barracks_thumb.png',
    category: 'military',
  },
  
  ARCHERY_RANGE: {
    id: 'archery_range',
    name: 'Archery Range',
    description: 'Train ranged units',
    cost: { gold: 200, wood: 200 },
    buildTime: 50,
    maxHealth: 1000,
    thumbnailPath: '/buildings/archery_range_thumb.png',
    category: 'military',
  },
  
  FARM: {
    id: 'farm',
    name: 'Farm',
    description: 'Generates food over time',
    cost: { gold: 50, wood: 60 },
    buildTime: 30,
    maxHealth: 400,
    thumbnailPath: '/buildings/farm_thumb.png',
    category: 'economic',
  },
};

export const getBuildingType = (id: string): BuildingType | undefined => {
  return Object.values(BUILDING_TYPES).find(type => type.id === id);
};
