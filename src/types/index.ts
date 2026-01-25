// Resource types for the RTS game

export interface Position {
  x: number;
  y: number;
}

export interface Resources {
  wood: number;
  gold: number;
  stone: number;
  food: number;
}

export interface Tree {
  id: string;
  position: Position;
  woodAmount: number;
  maxWood: number;
  isGrowing: boolean;
  lastHarvestTime: number;
  isDepleted: boolean;
}

export interface Worker {
  id: string;
  position: Position;
  isGathering: boolean;
  targetTreeId: string | null;
  carryingWood: number;
  maxCarryCapacity: number;
}

export interface Building {
  id: string;
  type: 'lumber_camp' | 'town_hall' | 'barracks';
  position: Position;
  gatheringBonus: number;
}

export interface GameState {
  resources: Resources;
  trees: Tree[];
  workers: Worker[];
  buildings: Building[];
  gameSpeed: number;
  isPaused: boolean;
}

export const WOOD_PER_HARVEST = 10;
export const TREE_REGROWTH_TIME = 60000; // 60 seconds in ms
export const TREE_REGROWTH_RATE = 2; // wood per tick
export const LUMBER_CAMP_BONUS = 1.5; // 50% gathering bonus
export const WORKER_CARRY_CAPACITY = 10;
