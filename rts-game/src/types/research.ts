export type ResearchTech =
  | 'iron_weapons'
  | 'steel_armor'
  | 'advanced_metallurgy'
  | 'weapon_sharpening'
  | 'armor_reinforcement'
  | 'blacksmithing_mastery';

export interface ResearchItem {
  id: ResearchTech;
  name: string;
  description: string;
  icon?: string;
  cost: {
    gold: number;
    food?: number;
  };
  researchTime: number; // in seconds
  requirements?: ResearchTech[];
  effects: {
    description: string;
    modifier: {
      type: 'attack' | 'defense' | 'production_speed' | 'cost_reduction';
      value: number;
      unit?: string;
    };
  };
  building: 'blacksmith';
}

export interface ResearchProgress {
  techId: ResearchTech;
  progress: number; // 0-100
  startTime: number;
}
