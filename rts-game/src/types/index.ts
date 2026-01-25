export interface Position {
  x: number;
  y: number;
}

export interface Resource {
  id: string;
  type: 'gold' | 'wood' | 'stone';
  position: Position;
  amount: number;
}

export interface Unit {
  id: string;
  type: 'worker' | 'soldier';
  position: Position;
  isGathering: boolean;
  targetResource?: string;
  ownerId: string;
}

export interface RallyPoint {
  position: Position;
  targetResource?: Resource;
  isResourceRallyPoint: boolean;
}

export interface Building {
  id: string;
  type: 'town_hall' | 'barracks' | 'farm';
  position: Position;
  rallyPoint?: RallyPoint;
  ownerId: string;
  productionQueue: string[];
}

export interface GameState {
  buildings: Map<string, Building>;
  units: Map<string, Unit>;
  resources: Map<string, Resource>;
  players: Map<string, Player>;
}

export interface Player {
  id: string;
  name: string;
  gold: number;
  wood: number;
  stone: number;
}
