// Core game type definitions for Orca RTS

export interface Position {
  x: number;
  y: number;
  z: number;
}

export interface BuildingType {
  id: string;
  name: string;
  description: string;
  cost: ResourceCost;
  buildTime: number;
  maxHealth: number;
  modelPath?: string;
  thumbnailPath?: string;
  category: 'military' | 'economic' | 'research' | 'defense';
}

export interface ResourceCost {
  gold?: number;
  wood?: number;
  stone?: number;
  food?: number;
}

export interface Building {
  id: string;
  type: BuildingType;
  position: Position;
  health: number;
  isConstructing: boolean;
  constructionProgress: number;
  ownerId: string;
}

export interface ResearchTech {
  id: string;
  name: string;
  description: string;
  cost: ResourceCost;
  researchTime: number;
  prerequisites: string[];
  buildingRequired: string;
  effects: TechEffect[];
  icon?: string;
}

export interface TechEffect {
  type: 'stat_boost' | 'unlock_unit' | 'unlock_building' | 'resource_bonus';
  target?: string;
  value: number | string;
  description: string;
}

export interface ResearchProgress {
  techId: string;
  progress: number;
  isComplete: boolean;
  startTime: number;
}

export interface GameResources {
  gold: number;
  wood: number;
  stone: number;
  food: number;
}

export interface Player {
  id: string;
  name: string;
  resources: GameResources;
  buildings: Building[];
  completedResearch: string[];
  activeResearch: ResearchProgress | null;
}
