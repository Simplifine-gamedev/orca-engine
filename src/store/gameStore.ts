/**
 * Game Store - Central state management for Orca RTS
 * Manages resources, buildings, units, and game progression
 */

export interface Resources {
  gold: number;
  wood: number;
  stone: number;
  food: number;
}

export interface Building {
  id: string;
  type: BuildingType;
  level: number;
  position: { x: number; y: number };
  productionQueue: string[];
}

export interface Unit {
  id: string;
  type: UnitType;
  level: number;
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  attack: number;
  defense: number;
  isHero: boolean;
}

export enum BuildingType {
  TOWN_CENTER = 'town_center',
  BARRACKS = 'barracks',
  ARCHERY_RANGE = 'archery_range',
  STABLE = 'stable',
  WORKSHOP = 'workshop',
  TEMPLE = 'temple',
  MARKET = 'market',
  BLACKSMITH = 'blacksmith',
  ACADEMY = 'academy',
  DEFENSE_TOWER = 'defense_tower',
}

export enum UnitType {
  WORKER = 'worker',
  SWORDSMAN = 'swordsman',
  ARCHER = 'archer',
  CAVALRY = 'cavalry',
  SIEGE_ENGINE = 'siege_engine',
  MAGE = 'mage',
  HERO_WARRIOR = 'hero_warrior',
  HERO_ARCHER = 'hero_archer',
  HERO_MAGE = 'hero_mage',
}

export interface BuildingUpgrade {
  buildingType: BuildingType;
  level: number;
  cost: Resources;
  benefits: string[];
  unlocks?: BuildingType[];
}

export interface UnitUpgrade {
  unitType: UnitType;
  upgradeType: 'armor' | 'weapon' | 'health' | 'speed' | 'special';
  level: number;
  cost: Resources;
  effect: {
    attack?: number;
    defense?: number;
    health?: number;
    speed?: number;
  };
}

export interface GameState {
  resources: Resources;
  buildings: Building[];
  units: Unit[];
  population: {
    current: number;
    max: number;
  };
  buildingUpgrades: Map<string, number>; // buildingType-level -> level
  unitUpgrades: Map<string, number>; // unitType-upgradeType-level -> level
}

// Building upgrade definitions
export const BUILDING_UPGRADES: BuildingUpgrade[] = [
  // Town Center upgrades
  {
    buildingType: BuildingType.TOWN_CENTER,
    level: 2,
    cost: { gold: 500, wood: 300, stone: 200, food: 0 },
    benefits: ['+5 population cap', 'Unlock Barracks', 'Unlock Market'],
    unlocks: [BuildingType.BARRACKS, BuildingType.MARKET],
  },
  {
    buildingType: BuildingType.TOWN_CENTER,
    level: 3,
    cost: { gold: 1000, wood: 600, stone: 500, food: 0 },
    benefits: ['+10 population cap', 'Unlock Blacksmith', '+20% resource generation'],
    unlocks: [BuildingType.BLACKSMITH],
  },
  {
    buildingType: BuildingType.TOWN_CENTER,
    level: 4,
    cost: { gold: 2000, wood: 1200, stone: 1000, food: 0 },
    benefits: ['+15 population cap', 'Unlock Academy', 'Unlock Temple', '+50% resource generation'],
    unlocks: [BuildingType.ACADEMY, BuildingType.TEMPLE],
  },
  {
    buildingType: BuildingType.TOWN_CENTER,
    level: 5,
    cost: { gold: 5000, wood: 3000, stone: 2500, food: 0 },
    benefits: ['+20 population cap', 'Unlock Hero units', '+100% resource generation', 'Special abilities'],
    unlocks: [],
  },

  // Barracks upgrades
  {
    buildingType: BuildingType.BARRACKS,
    level: 2,
    cost: { gold: 300, wood: 200, stone: 100, food: 0 },
    benefits: ['Unlock advanced infantry', 'Faster training time'],
  },
  {
    buildingType: BuildingType.BARRACKS,
    level: 3,
    cost: { gold: 800, wood: 500, stone: 300, food: 0 },
    benefits: ['Unlock elite infantry', 'Champion training available'],
  },

  // Blacksmith upgrades
  {
    buildingType: BuildingType.BLACKSMITH,
    level: 2,
    cost: { gold: 400, wood: 300, stone: 200, food: 0 },
    benefits: ['Unlock advanced unit upgrades', 'Reduced upgrade costs'],
  },
  {
    buildingType: BuildingType.BLACKSMITH,
    level: 3,
    cost: { gold: 1000, wood: 700, stone: 500, food: 0 },
    benefits: ['Unlock legendary equipment', 'Unit upgrade bonuses doubled'],
  },

  // Academy upgrades
  {
    buildingType: BuildingType.ACADEMY,
    level: 2,
    cost: { gold: 1500, wood: 500, stone: 800, food: 0 },
    benefits: ['Unlock advanced magic', 'Mage units available'],
  },
];

// Unit upgrade definitions
export const UNIT_UPGRADES: UnitUpgrade[] = [
  // Swordsman upgrades
  {
    unitType: UnitType.SWORDSMAN,
    upgradeType: 'armor',
    level: 1,
    cost: { gold: 200, wood: 0, stone: 100, food: 50 },
    effect: { defense: 2 },
  },
  {
    unitType: UnitType.SWORDSMAN,
    upgradeType: 'armor',
    level: 2,
    cost: { gold: 400, wood: 0, stone: 200, food: 100 },
    effect: { defense: 4 },
  },
  {
    unitType: UnitType.SWORDSMAN,
    upgradeType: 'weapon',
    level: 1,
    cost: { gold: 250, wood: 0, stone: 150, food: 50 },
    effect: { attack: 3 },
  },
  {
    unitType: UnitType.SWORDSMAN,
    upgradeType: 'weapon',
    level: 2,
    cost: { gold: 500, wood: 0, stone: 300, food: 100 },
    effect: { attack: 6 },
  },

  // Archer upgrades
  {
    unitType: UnitType.ARCHER,
    upgradeType: 'armor',
    level: 1,
    cost: { gold: 150, wood: 100, stone: 50, food: 50 },
    effect: { defense: 1 },
  },
  {
    unitType: UnitType.ARCHER,
    upgradeType: 'weapon',
    level: 1,
    cost: { gold: 200, wood: 150, stone: 0, food: 50 },
    effect: { attack: 4 },
  },
  {
    unitType: UnitType.ARCHER,
    upgradeType: 'weapon',
    level: 2,
    cost: { gold: 400, wood: 300, stone: 0, food: 100 },
    effect: { attack: 8 },
  },

  // Cavalry upgrades
  {
    unitType: UnitType.CAVALRY,
    upgradeType: 'armor',
    level: 1,
    cost: { gold: 300, wood: 0, stone: 200, food: 100 },
    effect: { defense: 3 },
  },
  {
    unitType: UnitType.CAVALRY,
    upgradeType: 'weapon',
    level: 1,
    cost: { gold: 350, wood: 0, stone: 250, food: 100 },
    effect: { attack: 5 },
  },
  {
    unitType: UnitType.CAVALRY,
    upgradeType: 'speed',
    level: 1,
    cost: { gold: 200, wood: 100, stone: 0, food: 150 },
    effect: { speed: 20 },
  },

  // Mage upgrades
  {
    unitType: UnitType.MAGE,
    upgradeType: 'special',
    level: 1,
    cost: { gold: 500, wood: 0, stone: 0, food: 200 },
    effect: { attack: 10 },
  },
];

// Store implementation
class GameStore {
  private state: GameState = {
    resources: {
      gold: 1000,
      wood: 500,
      stone: 300,
      food: 200,
    },
    buildings: [
      {
        id: 'tc1',
        type: BuildingType.TOWN_CENTER,
        level: 1,
        position: { x: 0, y: 0 },
        productionQueue: [],
      },
    ],
    units: [],
    population: {
      current: 5,
      max: 10,
    },
    buildingUpgrades: new Map(),
    unitUpgrades: new Map(),
  };

  private listeners: Set<(state: GameState) => void> = new Set();

  getState(): GameState {
    return this.state;
  }

  subscribe(listener: (state: GameState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.state));
  }

  // Resource management
  addResources(resources: Partial<Resources>): void {
    this.state.resources = {
      gold: this.state.resources.gold + (resources.gold || 0),
      wood: this.state.resources.wood + (resources.wood || 0),
      stone: this.state.resources.stone + (resources.stone || 0),
      food: this.state.resources.food + (resources.food || 0),
    };
    this.notifyListeners();
  }

  canAfford(cost: Resources): boolean {
    return (
      this.state.resources.gold >= cost.gold &&
      this.state.resources.wood >= cost.wood &&
      this.state.resources.stone >= cost.stone &&
      this.state.resources.food >= cost.food
    );
  }

  spendResources(cost: Resources): boolean {
    if (!this.canAfford(cost)) {
      return false;
    }
    this.state.resources = {
      gold: this.state.resources.gold - cost.gold,
      wood: this.state.resources.wood - cost.wood,
      stone: this.state.resources.stone - cost.stone,
      food: this.state.resources.food - cost.food,
    };
    this.notifyListeners();
    return true;
  }

  // Building management
  upgradeBuilding(buildingId: string): boolean {
    const building = this.state.buildings.find(b => b.id === buildingId);
    if (!building) return false;

    const upgrade = BUILDING_UPGRADES.find(
      u => u.buildingType === building.type && u.level === building.level + 1
    );

    if (!upgrade || !this.spendResources(upgrade.cost)) {
      return false;
    }

    building.level++;
    const key = `${building.type}-${building.level}`;
    this.state.buildingUpgrades.set(key, building.level);

    // Update population cap if town center
    if (building.type === BuildingType.TOWN_CENTER) {
      const popIncrease = parseInt(upgrade.benefits[0].match(/\+(\d+)/)?.[1] || '0');
      this.state.population.max += popIncrease;
    }

    this.notifyListeners();
    return true;
  }

  canBuildBuilding(buildingType: BuildingType): boolean {
    // Check if building is unlocked
    const requiredUpgrades = BUILDING_UPGRADES.filter(u => 
      u.unlocks?.includes(buildingType)
    );
    
    if (requiredUpgrades.length === 0 && buildingType !== BuildingType.TOWN_CENTER) {
      return false;
    }

    for (const upgrade of requiredUpgrades) {
      const key = `${upgrade.buildingType}-${upgrade.level}`;
      if (this.state.buildingUpgrades.get(key) !== upgrade.level) {
        return false;
      }
    }

    return true;
  }

  // Unit management
  upgradeUnit(unitType: UnitType, upgradeType: UnitUpgrade['upgradeType'], level: number): boolean {
    const upgrade = UNIT_UPGRADES.find(
      u => u.unitType === unitType && u.upgradeType === upgradeType && u.level === level
    );

    if (!upgrade || !this.spendResources(upgrade.cost)) {
      return false;
    }

    const key = `${unitType}-${upgradeType}-${level}`;
    this.state.unitUpgrades.set(key, level);

    // Apply upgrade to all existing units of this type
    this.state.units
      .filter(unit => unit.type === unitType)
      .forEach(unit => {
        if (upgrade.effect.attack) unit.attack += upgrade.effect.attack;
        if (upgrade.effect.defense) unit.defense += upgrade.effect.defense;
        if (upgrade.effect.health) {
          unit.maxHealth += upgrade.effect.health;
          unit.health += upgrade.effect.health;
        }
      });

    this.notifyListeners();
    return true;
  }

  hasUnitUpgrade(unitType: UnitType, upgradeType: UnitUpgrade['upgradeType'], level: number): boolean {
    const key = `${unitType}-${upgradeType}-${level}`;
    return this.state.unitUpgrades.has(key);
  }

  // Hero unit creation
  createHeroUnit(heroType: UnitType): boolean {
    if (!heroType.startsWith('hero_')) {
      return false;
    }

    // Requires town center level 5
    const townCenter = this.state.buildings.find(b => b.type === BuildingType.TOWN_CENTER);
    if (!townCenter || townCenter.level < 5) {
      return false;
    }

    const heroCost: Resources = { gold: 2000, wood: 0, stone: 0, food: 500 };
    if (!this.spendResources(heroCost)) {
      return false;
    }

    const hero: Unit = {
      id: `hero_${Date.now()}`,
      type: heroType,
      level: 1,
      position: { x: 0, y: 0 },
      health: 500,
      maxHealth: 500,
      attack: 50,
      defense: 30,
      isHero: true,
    };

    this.state.units.push(hero);
    this.notifyListeners();
    return true;
  }
}

export const gameStore = new GameStore();
