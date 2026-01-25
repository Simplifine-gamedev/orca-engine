// Research system types for late-game resource sinks

export enum ResearchCategory {
  MILITARY = 'military',
  ECONOMY = 'economy',
  TECHNOLOGY = 'technology',
  MAGIC = 'magic',
  DEFENSE = 'defense',
}

export enum ResearchStatus {
  LOCKED = 'locked',
  AVAILABLE = 'available',
  RESEARCHING = 'researching',
  COMPLETED = 'completed',
}

export interface ResearchCost {
  gold: number;
  wood?: number;
  stone?: number;
  food?: number;
  mana?: number;
}

export interface Research {
  id: string;
  name: string;
  description: string;
  category: ResearchCategory;
  cost: ResearchCost;
  researchTime: number; // in seconds
  prerequisites: string[]; // IDs of required research
  effects: ResearchEffect[];
  icon?: string;
}

export interface ResearchEffect {
  type: EffectType;
  target: string; // unit type, building type, or 'global'
  value: number;
  description: string;
}

export enum EffectType {
  // Unit upgrades
  ATTACK_DAMAGE = 'attack_damage',
  ATTACK_SPEED = 'attack_speed',
  ARMOR = 'armor',
  MOVEMENT_SPEED = 'movement_speed',
  HP = 'hp',
  VISION_RANGE = 'vision_range',
  
  // Economic upgrades
  GATHERING_SPEED = 'gathering_speed',
  RESOURCE_CAPACITY = 'resource_capacity',
  BUILDING_SPEED = 'building_speed',
  
  // Special abilities
  UNLOCK_ABILITY = 'unlock_ability',
  UNLOCK_UNIT = 'unlock_unit',
  UNLOCK_BUILDING = 'unlock_building',
  
  // Other
  COST_REDUCTION = 'cost_reduction',
  TRAINING_TIME = 'training_time',
}

export interface ResearchProgress {
  researchId: string;
  startTime: number;
  totalTime: number;
  progress: number; // 0-1
}
