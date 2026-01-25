// Unit system types

export enum UnitType {
  WORKER = 'worker',
  WARRIOR = 'warrior',
  ARCHER = 'archer',
  CAVALRY = 'cavalry',
  SIEGE = 'siege',
  MAGE = 'mage',
  HERO = 'hero',
  PRIEST = 'priest',
}

export interface UnitCost {
  gold: number;
  food: number;
  wood?: number;
  stone?: number;
}

export interface Unit {
  id: string;
  type: UnitType;
  name: string;
  description: string;
  baseCost: UnitCost;
  trainingTime: number; // in seconds
  baseStats: UnitStats;
  upgradeLevel: number;
  maxUpgradeLevel: number;
  prerequisites: UnitPrerequisite[];
  abilities: UnitAbility[];
}

export interface UnitStats {
  hp: number;
  attackDamage: number;
  attackSpeed: number; // attacks per second
  armor: number;
  movementSpeed: number;
  visionRange: number;
  attackRange: number;
}

export interface UnitPrerequisite {
  buildingType?: string;
  buildingLevel?: number;
  researchId?: string;
}

export interface UnitAbility {
  id: string;
  name: string;
  description: string;
  cooldown: number; // in seconds
  manaCost?: number;
  unlockResearchId?: string;
}

export interface UnitUpgrade {
  id: string;
  unitType: UnitType;
  level: number;
  name: string;
  description: string;
  cost: UnitCost;
  researchTime: number;
  statBonuses: Partial<UnitStats>;
  prerequisiteResearch: string[];
}

export interface UnitInstance {
  id: string;
  unitId: string;
  position: { x: number; y: number };
  health: number;
  currentStats: UnitStats;
  experience: number;
  level: number;
  abilityStates: Map<string, AbilityState>;
}

export interface AbilityState {
  abilityId: string;
  cooldownRemaining: number;
  isActive: boolean;
}
