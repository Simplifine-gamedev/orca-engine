// Building system types

export enum BuildingType {
  TOWN_CENTER = 'town_center',
  BARRACKS = 'barracks',
  ARCHERY_RANGE = 'archery_range',
  STABLE = 'stable',
  WORKSHOP = 'workshop',
  FARM = 'farm',
  LUMBER_MILL = 'lumber_mill',
  STONE_MINE = 'stone_mine',
  GOLD_MINE = 'gold_mine',
  TOWER = 'tower',
  WALL = 'wall',
  BLACKSMITH = 'blacksmith',
  TEMPLE = 'temple',
  MARKET = 'market',
  ACADEMY = 'academy',
}

export interface BuildingCost {
  gold: number;
  wood?: number;
  stone?: number;
  food?: number;
}

export interface Building {
  id: string;
  type: BuildingType;
  name: string;
  description: string;
  level: number;
  maxLevel: number;
  baseCost: BuildingCost;
  buildTime: number; // in seconds
  upgradeCost: BuildingCost;
  upgradeTime: number;
  prerequisites: BuildingPrerequisite[];
  effects: BuildingEffect[];
  unlocksAt: BuildingPrerequisite[];
}

export interface BuildingPrerequisite {
  buildingType?: BuildingType;
  level?: number;
  researchId?: string;
}

export interface BuildingEffect {
  type: string;
  value: number;
  description: string;
}

export interface BuildingInstance {
  id: string;
  buildingId: string;
  position: { x: number; y: number };
  level: number;
  isUpgrading: boolean;
  upgradeProgress: number;
  health: number;
  maxHealth: number;
}
