export interface UnitConfig {
  name: string;
  type: string;
  description: string;
  cost: {
    wood?: number;
    stone?: number;
    gold?: number;
    food?: number;
  };
  trainTime: number;
  health: number;
  attack: number;
  defense: number;
  speed: number;
  range: number;
  attackSpeed: number;
  trainingBuilding: string;
}

export interface FactionConfig {
  id: string;
  name: string;
  description: string;
  color: string;
  startingResources: {
    wood: number;
    stone: number;
    gold: number;
    food: number;
  };
  availableBuildings: string[];
  units: Record<string, UnitConfig>;
  bonuses?: {
    description: string;
    effect: string;
  }[];
}

export const unitConfigs: Record<string, UnitConfig> = {
  worker: {
    name: 'Worker',
    type: 'worker',
    description: 'Gathers resources and constructs buildings',
    cost: { food: 50 },
    trainTime: 15,
    health: 40,
    attack: 3,
    defense: 0,
    speed: 1.2,
    range: 0,
    attackSpeed: 1.0,
    trainingBuilding: 'town_hall',
  },

  warrior: {
    name: 'Warrior',
    type: 'warrior',
    description: 'Basic melee infantry unit',
    cost: { food: 60, gold: 20 },
    trainTime: 20,
    health: 100,
    attack: 12,
    defense: 2,
    speed: 1.0,
    range: 0,
    attackSpeed: 1.5,
    trainingBuilding: 'barracks',
  },

  swordsman: {
    name: 'Swordsman',
    type: 'swordsman',
    description: 'Advanced melee infantry unit',
    cost: { food: 75, gold: 35 },
    trainTime: 25,
    health: 130,
    attack: 16,
    defense: 4,
    speed: 0.95,
    range: 0,
    attackSpeed: 1.4,
    trainingBuilding: 'barracks',
  },

  knight: {
    name: 'Knight',
    type: 'knight',
    description: 'Elite heavily armored melee unit',
    cost: { food: 100, gold: 60 },
    trainTime: 35,
    health: 180,
    attack: 20,
    defense: 8,
    speed: 0.9,
    range: 0,
    attackSpeed: 1.3,
    trainingBuilding: 'barracks',
  },

  archer: {
    name: 'Archer',
    type: 'archer',
    description: 'Basic ranged unit trained at the archery range',
    cost: { food: 50, wood: 40 },
    trainTime: 22,
    health: 60,
    attack: 10,
    defense: 1,
    speed: 1.1,
    range: 6,
    attackSpeed: 2.0,
    trainingBuilding: 'archery_range',
  },

  crossbowman: {
    name: 'Crossbowman',
    type: 'crossbowman',
    description: 'Advanced ranged unit with high damage and range',
    cost: { food: 70, wood: 50, gold: 20 },
    trainTime: 30,
    health: 70,
    attack: 16,
    defense: 2,
    speed: 1.0,
    range: 7,
    attackSpeed: 1.6,
    trainingBuilding: 'archery_range',
  },

  cavalry: {
    name: 'Cavalry',
    type: 'cavalry',
    description: 'Fast mounted melee unit',
    cost: { food: 100, gold: 50 },
    trainTime: 30,
    health: 150,
    attack: 18,
    defense: 3,
    speed: 1.6,
    range: 0,
    attackSpeed: 1.8,
    trainingBuilding: 'stable',
  },

  horse_archer: {
    name: 'Horse Archer',
    type: 'horse_archer',
    description: 'Fast mounted ranged unit',
    cost: { food: 90, wood: 60, gold: 40 },
    trainTime: 35,
    health: 110,
    attack: 12,
    defense: 2,
    speed: 1.5,
    range: 5,
    attackSpeed: 2.2,
    trainingBuilding: 'stable',
  },
};

export const factions: Record<string, FactionConfig> = {
  humans: {
    id: 'humans',
    name: 'Human Alliance',
    description: 'Balanced faction with versatile units and buildings',
    color: '#4169e1',
    startingResources: {
      wood: 500,
      stone: 250,
      gold: 250,
      food: 500,
    },
    availableBuildings: [
      'town_hall',
      'barracks',
      'archery_range',
      'stable',
      'blacksmith',
      'tower',
      'farm',
      'lumber_mill',
      'stone_mine',
      'market',
    ],
    units: {
      worker: unitConfigs.worker,
      warrior: unitConfigs.warrior,
      swordsman: unitConfigs.swordsman,
      knight: unitConfigs.knight,
      archer: unitConfigs.archer,
      crossbowman: unitConfigs.crossbowman,
      cavalry: unitConfigs.cavalry,
    },
    bonuses: [
      {
        description: 'Workers gather resources 10% faster',
        effect: 'worker_gather_speed:1.1',
      },
      {
        description: 'Buildings cost 5% less wood',
        effect: 'building_wood_cost:0.95',
      },
    ],
  },

  elves: {
    id: 'elves',
    name: 'Elven Kingdom',
    description: 'Ranged combat specialists with superior archers',
    color: '#228b22',
    startingResources: {
      wood: 600,
      stone: 200,
      gold: 250,
      food: 450,
    },
    availableBuildings: [
      'town_hall',
      'barracks',
      'archery_range',
      'stable',
      'blacksmith',
      'tower',
      'farm',
      'lumber_mill',
      'market',
    ],
    units: {
      worker: unitConfigs.worker,
      warrior: unitConfigs.warrior,
      archer: {
        ...unitConfigs.archer,
        attack: 12,
        range: 7,
        trainTime: 18,
      },
      crossbowman: {
        ...unitConfigs.crossbowman,
        attack: 18,
        range: 8,
        trainTime: 25,
      },
      horse_archer: unitConfigs.horse_archer,
    },
    bonuses: [
      {
        description: 'Archers and crossbowmen have +2 range and +2 attack',
        effect: 'archer_range:+2,archer_attack:+2',
      },
      {
        description: 'Archery range trains units 20% faster',
        effect: 'archery_range_train_speed:0.8',
      },
      {
        description: 'Units move 10% faster in forests',
        effect: 'forest_movement_speed:1.1',
      },
    ],
  },

  orcs: {
    id: 'orcs',
    name: 'Orcish Horde',
    description: 'Aggressive faction focused on melee combat',
    color: '#8b0000',
    startingResources: {
      wood: 450,
      stone: 300,
      gold: 200,
      food: 550,
    },
    availableBuildings: [
      'town_hall',
      'barracks',
      'archery_range',
      'stable',
      'blacksmith',
      'tower',
      'farm',
      'lumber_mill',
      'stone_mine',
    ],
    units: {
      worker: unitConfigs.worker,
      warrior: {
        ...unitConfigs.warrior,
        health: 120,
        attack: 15,
        trainTime: 18,
      },
      swordsman: {
        ...unitConfigs.swordsman,
        health: 150,
        attack: 19,
      },
      knight: {
        ...unitConfigs.knight,
        health: 210,
        attack: 24,
      },
      archer: unitConfigs.archer,
      cavalry: unitConfigs.cavalry,
    },
    bonuses: [
      {
        description: 'Melee units have +20% health and attack',
        effect: 'melee_health:1.2,melee_attack:1.2',
      },
      {
        description: 'Barracks trains units 15% faster',
        effect: 'barracks_train_speed:0.85',
      },
      {
        description: 'Buildings cost 10% less stone',
        effect: 'building_stone_cost:0.9',
      },
    ],
  },

  undead: {
    id: 'undead',
    name: 'Undead Legion',
    description: 'Dark faction with cheap, expendable units',
    color: '#4b0082',
    startingResources: {
      wood: 400,
      stone: 350,
      gold: 300,
      food: 450,
    },
    availableBuildings: [
      'town_hall',
      'barracks',
      'archery_range',
      'blacksmith',
      'tower',
      'stone_mine',
      'market',
    ],
    units: {
      worker: {
        ...unitConfigs.worker,
        cost: { food: 40 },
        health: 35,
      },
      warrior: {
        ...unitConfigs.warrior,
        cost: { food: 50, gold: 15 },
        health: 90,
      },
      swordsman: unitConfigs.swordsman,
      archer: {
        ...unitConfigs.archer,
        cost: { food: 45, wood: 35 },
      },
      crossbowman: unitConfigs.crossbowman,
    },
    bonuses: [
      {
        description: 'Units cost 15% less to train',
        effect: 'unit_cost:0.85',
      },
      {
        description: 'Units can be trained 25% faster',
        effect: 'unit_train_speed:0.75',
      },
      {
        description: 'Buildings have +50% health',
        effect: 'building_health:1.5',
      },
    ],
  },
};

export function getFaction(factionId: string): FactionConfig | undefined {
  return factions[factionId];
}

export function getUnitConfig(unitType: string, factionId?: string): UnitConfig | undefined {
  if (factionId) {
    const faction = factions[factionId];
    return faction?.units[unitType];
  }
  return unitConfigs[unitType];
}

export function canTrainUnit(
  unitType: string,
  buildingType: string,
  resources: { wood?: number; stone?: number; gold?: number; food?: number },
  factionId?: string
): { canTrain: boolean; reason?: string } {
  const unitConfig = getUnitConfig(unitType, factionId);
  
  if (!unitConfig) {
    return { canTrain: false, reason: 'Unit type not found' };
  }

  if (unitConfig.trainingBuilding !== buildingType) {
    return {
      canTrain: false,
      reason: `${unitConfig.name} cannot be trained at this building`,
    };
  }

  if (
    (unitConfig.cost.wood || 0) > (resources.wood || 0) ||
    (unitConfig.cost.stone || 0) > (resources.stone || 0) ||
    (unitConfig.cost.gold || 0) > (resources.gold || 0) ||
    (unitConfig.cost.food || 0) > (resources.food || 0)
  ) {
    return { canTrain: false, reason: 'Insufficient resources' };
  }

  return { canTrain: true };
}

export default factions;
