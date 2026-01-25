// Core game types for RTS

export interface Resources {
  gold: number;
  wood: number;
  stone: number;
  food: number;
}

export interface ResourceIncome {
  gold: number;
  wood: number;
  stone: number;
  food: number;
}

export interface ResourceCost {
  gold?: number;
  wood?: number;
  stone?: number;
  food?: number;
}

export interface Building {
  id: string;
  name: string;
  type: string;
  cost: ResourceCost;
  buildTime: number;
  produces?: Partial<ResourceIncome>;
  description: string;
  icon: string;
}

export interface Unit {
  id: string;
  name: string;
  type: string;
  cost: ResourceCost;
  buildTime: number;
  description: string;
  icon: string;
}

export interface GameState {
  resources: Resources;
  income: ResourceIncome;
  buildings: Building[];
  units: Unit[];
  selectedBuilding: Building | null;
}

export const BUILDING_TYPES: Record<string, Building> = {
  goldMine: {
    id: 'goldMine',
    name: 'Gold Mine',
    type: 'resource',
    cost: { wood: 100, stone: 50 },
    buildTime: 30,
    produces: { gold: 5 },
    description: 'Produces gold over time. Used for training units and building structures.',
    icon: '⛏️'
  },
  lumberMill: {
    id: 'lumberMill',
    name: 'Lumber Mill',
    type: 'resource',
    cost: { gold: 75, stone: 25 },
    buildTime: 25,
    produces: { wood: 8 },
    description: 'Harvests wood. Required for most buildings and siege equipment.',
    icon: '🪓'
  },
  quarry: {
    id: 'quarry',
    name: 'Quarry',
    type: 'resource',
    cost: { gold: 100, wood: 75 },
    buildTime: 35,
    produces: { stone: 4 },
    description: 'Extracts stone. Used for defensive structures and advanced buildings.',
    icon: '🗿'
  },
  farm: {
    id: 'farm',
    name: 'Farm',
    type: 'resource',
    cost: { gold: 50, wood: 100 },
    buildTime: 20,
    produces: { food: 10 },
    description: 'Grows food. Required to sustain your army and population.',
    icon: '🌾'
  },
  barracks: {
    id: 'barracks',
    name: 'Barracks',
    type: 'military',
    cost: { gold: 150, wood: 200, stone: 100 },
    buildTime: 60,
    description: 'Trains infantry units. Required to build an army.',
    icon: '⚔️'
  },
  townHall: {
    id: 'townHall',
    name: 'Town Hall',
    type: 'main',
    cost: { gold: 500, wood: 300, stone: 200 },
    buildTime: 120,
    description: 'Main building. Allows advanced research and unit production.',
    icon: '🏛️'
  }
};

export const UNIT_TYPES: Record<string, Unit> = {
  worker: {
    id: 'worker',
    name: 'Worker',
    type: 'worker',
    cost: { food: 50 },
    buildTime: 10,
    description: 'Gathers resources and constructs buildings.',
    icon: '👷'
  },
  warrior: {
    id: 'warrior',
    name: 'Warrior',
    type: 'infantry',
    cost: { food: 60, gold: 40 },
    buildTime: 15,
    description: 'Basic melee unit. Strong against archers.',
    icon: '🗡️'
  },
  archer: {
    id: 'archer',
    name: 'Archer',
    type: 'ranged',
    cost: { food: 50, gold: 30, wood: 20 },
    buildTime: 12,
    description: 'Ranged unit. Effective against infantry.',
    icon: '🏹'
  },
  cavalry: {
    id: 'cavalry',
    name: 'Cavalry',
    type: 'mounted',
    cost: { food: 100, gold: 80 },
    buildTime: 25,
    description: 'Fast mounted unit. Good for raids and quick strikes.',
    icon: '🐎'
  }
};
