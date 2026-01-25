export interface BuildingModel {
  name: string;
  type: string;
  description: string;
  width: number;
  height: number;
  color: string;
  model?: string;
  cost: {
    wood?: number;
    stone?: number;
    gold?: number;
  };
  buildTime: number;
  trainableUnits?: string[];
  maxHealth: number;
  armor: number;
  requirements?: string[];
}

export const buildingModels: Record<string, BuildingModel> = {
  town_hall: {
    name: 'Town Hall',
    type: 'town_hall',
    description: 'Main building for resource collection and worker production',
    width: 96,
    height: 96,
    color: '#8b7355',
    cost: {
      wood: 0,
      stone: 0,
      gold: 0,
    },
    buildTime: 0,
    trainableUnits: ['worker'],
    maxHealth: 2000,
    armor: 5,
  },

  barracks: {
    name: 'Barracks',
    type: 'barracks',
    description: 'Trains melee infantry units',
    width: 80,
    height: 80,
    color: '#654321',
    cost: {
      wood: 150,
      stone: 0,
      gold: 0,
    },
    buildTime: 60,
    trainableUnits: ['warrior', 'swordsman', 'knight'],
    maxHealth: 1500,
    armor: 3,
    requirements: ['town_hall'],
  },

  archery_range: {
    name: 'Archery Range',
    type: 'archery_range',
    description: 'Trains ranged units including archers and crossbowmen',
    width: 80,
    height: 80,
    color: '#6b8e23',
    cost: {
      wood: 175,
      stone: 25,
      gold: 0,
    },
    buildTime: 70,
    trainableUnits: ['archer', 'crossbowman'],
    maxHealth: 1200,
    armor: 2,
    requirements: ['town_hall'],
  },

  stable: {
    name: 'Stable',
    type: 'stable',
    description: 'Trains cavalry units',
    width: 96,
    height: 80,
    color: '#8b6914',
    cost: {
      wood: 200,
      stone: 0,
      gold: 50,
    },
    buildTime: 80,
    trainableUnits: ['cavalry', 'horse_archer'],
    maxHealth: 1400,
    armor: 2,
    requirements: ['town_hall'],
  },

  blacksmith: {
    name: 'Blacksmith',
    type: 'blacksmith',
    description: 'Research upgrades for units',
    width: 64,
    height: 64,
    color: '#36454f',
    cost: {
      wood: 125,
      stone: 75,
      gold: 0,
    },
    buildTime: 50,
    maxHealth: 1000,
    armor: 4,
    requirements: ['town_hall'],
  },

  tower: {
    name: 'Guard Tower',
    type: 'tower',
    description: 'Defensive structure that attacks enemies',
    width: 48,
    height: 64,
    color: '#708090',
    cost: {
      wood: 50,
      stone: 100,
      gold: 0,
    },
    buildTime: 40,
    maxHealth: 800,
    armor: 5,
    requirements: ['town_hall'],
  },

  farm: {
    name: 'Farm',
    type: 'farm',
    description: 'Provides food resources',
    width: 64,
    height: 64,
    color: '#daa520',
    cost: {
      wood: 60,
      stone: 0,
      gold: 0,
    },
    buildTime: 30,
    maxHealth: 500,
    armor: 0,
    requirements: ['town_hall'],
  },

  lumber_mill: {
    name: 'Lumber Mill',
    type: 'lumber_mill',
    description: 'Improves wood gathering efficiency',
    width: 80,
    height: 64,
    color: '#8b4513',
    cost: {
      wood: 100,
      stone: 0,
      gold: 0,
    },
    buildTime: 40,
    maxHealth: 800,
    armor: 1,
    requirements: ['town_hall'],
  },

  stone_mine: {
    name: 'Stone Mine',
    type: 'stone_mine',
    description: 'Improves stone gathering efficiency',
    width: 80,
    height: 64,
    color: '#a9a9a9',
    cost: {
      wood: 75,
      stone: 50,
      gold: 0,
    },
    buildTime: 45,
    maxHealth: 1000,
    armor: 3,
    requirements: ['town_hall'],
  },

  market: {
    name: 'Market',
    type: 'market',
    description: 'Trade resources and improve economy',
    width: 80,
    height: 80,
    color: '#ffd700',
    cost: {
      wood: 100,
      stone: 50,
      gold: 50,
    },
    buildTime: 60,
    maxHealth: 1000,
    armor: 1,
    requirements: ['town_hall'],
  },
};

export function getBuildingModel(type: string): BuildingModel | undefined {
  return buildingModels[type];
}

export function getBuildingsByType(category: 'military' | 'economic' | 'defensive'): BuildingModel[] {
  const categories = {
    military: ['barracks', 'archery_range', 'stable'],
    economic: ['farm', 'lumber_mill', 'stone_mine', 'market'],
    defensive: ['tower'],
  };

  return categories[category]
    .map((type) => buildingModels[type])
    .filter((model): model is BuildingModel => model !== undefined);
}

export function canBuild(
  buildingType: string,
  constructedBuildings: string[],
  resources: { wood: number; stone: number; gold: number }
): { canBuild: boolean; reason?: string } {
  const model = buildingModels[buildingType];
  
  if (!model) {
    return { canBuild: false, reason: 'Building type not found' };
  }

  if (model.requirements) {
    for (const requirement of model.requirements) {
      if (!constructedBuildings.includes(requirement)) {
        const reqModel = buildingModels[requirement];
        return {
          canBuild: false,
          reason: `Requires ${reqModel?.name || requirement}`,
        };
      }
    }
  }

  if (
    (model.cost.wood || 0) > resources.wood ||
    (model.cost.stone || 0) > resources.stone ||
    (model.cost.gold || 0) > resources.gold
  ) {
    return { canBuild: false, reason: 'Insufficient resources' };
  }

  return { canBuild: true };
}

export default buildingModels;
