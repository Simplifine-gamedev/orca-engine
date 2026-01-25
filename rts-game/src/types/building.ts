export type BuildingType = 
  | 'headquarters'
  | 'barracks'
  | 'blacksmith'
  | 'archery_range'
  | 'stable'
  | 'farm'
  | 'mine'
  | 'lumbermill';

export interface BuildingModel {
  id: BuildingType;
  name: string;
  description: string;
  cost: {
    wood: number;
    stone: number;
    gold: number;
  };
  buildTime: number; // in seconds
  size: {
    width: number;
    height: number;
  };
  modelPath?: string;
  thumbnail?: string;
  hitPoints: number;
  canProduce?: string[];
  canResearch?: string[];
}

export interface PlacedBuilding extends BuildingModel {
  instanceId: string;
  position: {
    x: number;
    y: number;
  };
  health: number;
  productionQueue: string[];
  constructionProgress: number; // 0-100
}
