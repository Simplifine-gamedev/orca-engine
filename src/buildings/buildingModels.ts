/**
 * Building Models and Type Definitions for Orca RTS
 * Defines all building types, their properties, and model configurations
 */

export enum BuildingType {
  ARCHERY_RANGE = 'archery_range',
  BLACKSMITH = 'blacksmith',
  WALL = 'wall',
  BARRACKS = 'barracks',
  TOWN_CENTER = 'town_center',
}

export interface BuildingModel {
  type: BuildingType;
  name: string;
  width: number;
  height: number;
  depth: number;
  modelPath?: string;
  color: string;
  cost: {
    wood: number;
    stone: number;
    gold: number;
  };
  buildTime: number; // in seconds
  hasGhostPreview: boolean; // Whether to show ghost/blueprint preview
}

export const BUILDING_MODELS: Record<BuildingType, BuildingModel> = {
  [BuildingType.ARCHERY_RANGE]: {
    type: BuildingType.ARCHERY_RANGE,
    name: 'Archery Range',
    width: 4,
    height: 3,
    depth: 4,
    color: '#8B4513',
    cost: { wood: 150, stone: 50, gold: 0 },
    buildTime: 30,
    hasGhostPreview: true,
  },
  [BuildingType.BLACKSMITH]: {
    type: BuildingType.BLACKSMITH,
    name: 'Blacksmith',
    width: 3,
    height: 3,
    depth: 3,
    color: '#2C2C2C',
    cost: { wood: 100, stone: 100, gold: 50 },
    buildTime: 25,
    hasGhostPreview: true,
  },
  [BuildingType.WALL]: {
    type: BuildingType.WALL,
    name: 'Wall',
    width: 1,
    height: 2,
    depth: 1,
    color: '#808080',
    cost: { wood: 0, stone: 5, gold: 0 },
    buildTime: 5,
    hasGhostPreview: true,
  },
  [BuildingType.BARRACKS]: {
    type: BuildingType.BARRACKS,
    name: 'Barracks',
    width: 5,
    height: 3,
    depth: 4,
    color: '#654321',
    cost: { wood: 200, stone: 100, gold: 0 },
    buildTime: 40,
    hasGhostPreview: true,
  },
  [BuildingType.TOWN_CENTER]: {
    type: BuildingType.TOWN_CENTER,
    name: 'Town Center',
    width: 6,
    height: 4,
    depth: 6,
    color: '#DAA520',
    cost: { wood: 500, stone: 300, gold: 200 },
    buildTime: 60,
    hasGhostPreview: true,
  },
};

export interface BuildingPlacement {
  type: BuildingType;
  position: { x: number; y: number; z: number };
  rotation: number;
  isValid: boolean; // Whether placement is valid (not overlapping, etc.)
}

export function getBuildingModel(type: BuildingType): BuildingModel {
  return BUILDING_MODELS[type];
}
