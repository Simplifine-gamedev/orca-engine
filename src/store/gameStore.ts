import {
  Building,
  BuildingType,
  BuildingInstance,
  BuildingCost,
  BuildingPrerequisite,
} from '../types/buildings';
import {
  Unit,
  UnitType,
  UnitInstance,
  UnitUpgrade,
  UnitCost,
} from '../types/units';
import { ResearchStore } from './researchStore';

// Game Resources
export interface Resources {
  gold: number;
  wood: number;
  stone: number;
  food: number;
  mana: number;
}

// Game Store - Main game state management
export class GameStore {
  private resources: Resources;
  private buildings: Map<string, BuildingInstance> = new Map();
  private units: Map<string, UnitInstance> = new Map();
  private buildingDefinitions: Map<BuildingType, Building> = new Map();
  private unitDefinitions: Map<UnitType, Unit> = new Map();
  private unitUpgrades: Map<string, UnitUpgrade> = new Map();
  private researchStore: ResearchStore;
  private unlockedBuildings: Set<BuildingType> = new Set();
  private unlockedUnits: Set<UnitType> = new Set();

  constructor(researchStore: ResearchStore) {
    this.researchStore = researchStore;
    this.resources = {
      gold: 1000,
      wood: 500,
      stone: 500,
      food: 300,
      mana: 0,
    };
    this.initializeBuildingDefinitions();
    this.initializeUnitDefinitions();
    this.initializeUnitUpgrades();
    this.initializeStartingUnlocks();
  }

  // Initialize starting unlocks
  private initializeStartingUnlocks() {
    // Start with basic buildings
    this.unlockedBuildings.add(BuildingType.TOWN_CENTER);
    this.unlockedBuildings.add(BuildingType.BARRACKS);
    this.unlockedBuildings.add(BuildingType.FARM);
    this.unlockedBuildings.add(BuildingType.LUMBER_MILL);
    this.unlockedBuildings.add(BuildingType.STONE_MINE);
    this.unlockedBuildings.add(BuildingType.GOLD_MINE);

    // Start with basic units
    this.unlockedUnits.add(UnitType.WORKER);
    this.unlockedUnits.add(UnitType.WARRIOR);
    this.unlockedUnits.add(UnitType.ARCHER);
  }

  // Initialize building definitions
  private initializeBuildingDefinitions() {
    const buildings: Building[] = [
      {
        id: 'town_center',
        type: BuildingType.TOWN_CENTER,
        name: 'Town Center',
        description: 'Main building. Train workers and advance ages.',
        level: 1,
        maxLevel: 5,
        baseCost: { gold: 600, wood: 400, stone: 200 },
        buildTime: 120,
        upgradeCost: { gold: 800, wood: 500, stone: 300 },
        upgradeTime: 180,
        prerequisites: [],
        effects: [
          {
            type: 'population_capacity',
            value: 10,
            description: 'Increases population capacity by 10 per level',
          },
          {
            type: 'resource_generation',
            value: 1,
            description: 'Generates 1 gold per second per level',
          },
        ],
        unlocksAt: [],
      },
      {
        id: 'barracks',
        type: BuildingType.BARRACKS,
        name: 'Barracks',
        description: 'Train melee units',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 200, wood: 150 },
        buildTime: 60,
        upgradeCost: { gold: 400, wood: 300, stone: 100 },
        upgradeTime: 90,
        prerequisites: [],
        effects: [
          {
            type: 'training_speed',
            value: 1.1,
            description: '+10% training speed per level',
          },
        ],
        unlocksAt: [],
      },
      {
        id: 'archery_range',
        type: BuildingType.ARCHERY_RANGE,
        name: 'Archery Range',
        description: 'Train ranged units',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 200, wood: 175 },
        buildTime: 60,
        upgradeCost: { gold: 400, wood: 350, stone: 100 },
        upgradeTime: 90,
        prerequisites: [],
        effects: [
          {
            type: 'training_speed',
            value: 1.1,
            description: '+10% training speed per level',
          },
        ],
        unlocksAt: [],
      },
      {
        id: 'stable',
        type: BuildingType.STABLE,
        name: 'Stable',
        description: 'Train cavalry units',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 250, wood: 200, food: 100 },
        buildTime: 70,
        upgradeCost: { gold: 500, wood: 400, stone: 150, food: 200 },
        upgradeTime: 100,
        prerequisites: [{ buildingType: BuildingType.BARRACKS, level: 2 }],
        effects: [
          {
            type: 'training_speed',
            value: 1.15,
            description: '+15% training speed per level',
          },
        ],
        unlocksAt: [{ buildingType: BuildingType.BARRACKS, level: 2 }],
      },
      {
        id: 'workshop',
        type: BuildingType.WORKSHOP,
        name: 'Workshop',
        description: 'Train siege units and advanced machinery',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 400, wood: 300, stone: 200 },
        buildTime: 90,
        upgradeCost: { gold: 800, wood: 600, stone: 400 },
        upgradeTime: 120,
        prerequisites: [{ researchId: 'advanced_architecture' }],
        effects: [
          {
            type: 'siege_damage',
            value: 1.2,
            description: '+20% siege damage per level',
          },
        ],
        unlocksAt: [{ researchId: 'advanced_architecture' }],
      },
      {
        id: 'blacksmith',
        type: BuildingType.BLACKSMITH,
        name: 'Blacksmith',
        description: 'Research unit upgrades',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 200, wood: 100, stone: 100 },
        buildTime: 50,
        upgradeCost: { gold: 400, wood: 200, stone: 200 },
        upgradeTime: 75,
        prerequisites: [],
        effects: [
          {
            type: 'upgrade_discount',
            value: 0.1,
            description: '-10% upgrade cost per level',
          },
        ],
        unlocksAt: [],
      },
      {
        id: 'academy',
        type: BuildingType.ACADEMY,
        name: 'Academy',
        description: 'Research advanced technologies and magic',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 500, wood: 300, stone: 300 },
        buildTime: 100,
        upgradeCost: { gold: 1000, wood: 600, stone: 600 },
        upgradeTime: 150,
        prerequisites: [{ researchId: 'advanced_architecture' }],
        effects: [
          {
            type: 'research_speed',
            value: 1.15,
            description: '+15% research speed per level',
          },
          {
            type: 'mana_generation',
            value: 2,
            description: 'Generates 2 mana per second per level',
          },
        ],
        unlocksAt: [{ researchId: 'advanced_architecture' }],
      },
      {
        id: 'temple',
        type: BuildingType.TEMPLE,
        name: 'Temple',
        description: 'Generate mana and train priest units',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 400, wood: 250, stone: 250 },
        buildTime: 80,
        upgradeCost: { gold: 800, wood: 500, stone: 500 },
        upgradeTime: 120,
        prerequisites: [{ researchId: 'basic_magic' }],
        effects: [
          {
            type: 'mana_generation',
            value: 3,
            description: 'Generates 3 mana per second per level',
          },
        ],
        unlocksAt: [{ researchId: 'basic_magic' }],
      },
      {
        id: 'market',
        type: BuildingType.MARKET,
        name: 'Market',
        description: 'Trade resources and improve economy',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 300, wood: 200 },
        buildTime: 60,
        upgradeCost: { gold: 600, wood: 400, stone: 100 },
        upgradeTime: 90,
        prerequisites: [{ buildingType: BuildingType.TOWN_CENTER, level: 2 }],
        effects: [
          {
            type: 'trade_efficiency',
            value: 1.1,
            description: '+10% trade efficiency per level',
          },
        ],
        unlocksAt: [{ buildingType: BuildingType.TOWN_CENTER, level: 2 }],
      },
      {
        id: 'tower',
        type: BuildingType.TOWER,
        name: 'Defensive Tower',
        description: 'Defensive structure that attacks enemies',
        level: 1,
        maxLevel: 3,
        baseCost: { gold: 150, wood: 50, stone: 150 },
        buildTime: 40,
        upgradeCost: { gold: 300, wood: 100, stone: 300 },
        upgradeTime: 60,
        prerequisites: [{ researchId: 'fortification_1' }],
        effects: [
          {
            type: 'tower_damage',
            value: 1.3,
            description: '+30% damage per level',
          },
          {
            type: 'tower_range',
            value: 1.2,
            description: '+20% range per level',
          },
        ],
        unlocksAt: [{ researchId: 'fortification_1' }],
      },
    ];

    buildings.forEach((building) => {
      this.buildingDefinitions.set(building.type, building);
    });
  }

  // Initialize unit definitions
  private initializeUnitDefinitions() {
    const units: Unit[] = [
      {
        id: 'worker',
        type: UnitType.WORKER,
        name: 'Worker',
        description: 'Gathers resources and builds structures',
        baseCost: { gold: 50, food: 50 },
        trainingTime: 15,
        baseStats: {
          hp: 40,
          attackDamage: 3,
          attackSpeed: 0.5,
          armor: 0,
          movementSpeed: 1.0,
          visionRange: 6,
          attackRange: 1,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 0,
        prerequisites: [],
        abilities: [],
      },
      {
        id: 'warrior',
        type: UnitType.WARRIOR,
        name: 'Warrior',
        description: 'Basic melee unit with good armor',
        baseCost: { gold: 100, food: 50 },
        trainingTime: 20,
        baseStats: {
          hp: 100,
          attackDamage: 15,
          attackSpeed: 1.0,
          armor: 2,
          movementSpeed: 1.0,
          visionRange: 8,
          attackRange: 1,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 3,
        prerequisites: [],
        abilities: [
          {
            id: 'charge',
            name: 'Charge',
            description: 'Rush forward, dealing damage',
            cooldown: 15,
            unlockResearchId: 'melee_attack_2',
          },
        ],
      },
      {
        id: 'archer',
        type: UnitType.ARCHER,
        name: 'Archer',
        description: 'Ranged unit effective against unarmored targets',
        baseCost: { gold: 80, food: 40, wood: 30 },
        trainingTime: 18,
        baseStats: {
          hp: 60,
          attackDamage: 12,
          attackSpeed: 0.8,
          armor: 0,
          movementSpeed: 1.1,
          visionRange: 10,
          attackRange: 8,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 3,
        prerequisites: [],
        abilities: [
          {
            id: 'volley',
            name: 'Arrow Volley',
            description: 'Fire multiple arrows in an area',
            cooldown: 20,
            unlockResearchId: 'ranged_attack_2',
          },
        ],
      },
      {
        id: 'cavalry',
        type: UnitType.CAVALRY,
        name: 'Cavalry',
        description: 'Fast melee unit, good for hit-and-run',
        baseCost: { gold: 150, food: 100 },
        trainingTime: 25,
        baseStats: {
          hp: 120,
          attackDamage: 18,
          attackSpeed: 1.2,
          armor: 1,
          movementSpeed: 1.8,
          visionRange: 9,
          attackRange: 1,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 3,
        prerequisites: [{ buildingType: 'stable', buildingLevel: 1 }],
        abilities: [
          {
            id: 'trample',
            name: 'Trample',
            description: 'Charge through enemies',
            cooldown: 18,
            unlockResearchId: 'cavalry_speed',
          },
        ],
      },
      {
        id: 'siege',
        type: UnitType.SIEGE,
        name: 'Siege Engine',
        description: 'Powerful against buildings',
        baseCost: { gold: 300, wood: 200, stone: 100 },
        trainingTime: 60,
        baseStats: {
          hp: 80,
          attackDamage: 50,
          attackSpeed: 0.3,
          armor: 0,
          movementSpeed: 0.6,
          visionRange: 10,
          attackRange: 12,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 2,
        prerequisites: [{ researchId: 'ballistics' }],
        abilities: [],
      },
      {
        id: 'mage',
        type: UnitType.MAGE,
        name: 'Mage',
        description: 'Magical damage dealer',
        baseCost: { gold: 200, food: 50, mana: 50 },
        trainingTime: 30,
        baseStats: {
          hp: 50,
          attackDamage: 25,
          attackSpeed: 0.7,
          armor: 0,
          movementSpeed: 0.9,
          visionRange: 10,
          attackRange: 9,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 3,
        prerequisites: [{ researchId: 'basic_magic' }],
        abilities: [
          {
            id: 'fireball',
            name: 'Fireball',
            description: 'Launch a fireball at enemies',
            cooldown: 8,
            manaCost: 20,
          },
          {
            id: 'meteor',
            name: 'Meteor',
            description: 'Call down a devastating meteor',
            cooldown: 60,
            manaCost: 100,
            unlockResearchId: 'arcane_mastery',
          },
        ],
      },
      {
        id: 'hero',
        type: UnitType.HERO,
        name: 'Hero',
        description: 'Powerful unique unit',
        baseCost: { gold: 500, food: 200, mana: 100 },
        trainingTime: 90,
        baseStats: {
          hp: 300,
          attackDamage: 40,
          attackSpeed: 1.5,
          armor: 5,
          movementSpeed: 1.2,
          visionRange: 12,
          attackRange: 2,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 5,
        prerequisites: [{ researchId: 'hero_unit' }],
        abilities: [
          {
            id: 'warcry',
            name: 'War Cry',
            description: 'Buff nearby allies',
            cooldown: 30,
          },
          {
            id: 'heroic_strike',
            name: 'Heroic Strike',
            description: 'Deal massive damage',
            cooldown: 45,
          },
        ],
      },
      {
        id: 'priest',
        type: UnitType.PRIEST,
        name: 'Priest',
        description: 'Support unit with healing',
        baseCost: { gold: 150, food: 50, mana: 50 },
        trainingTime: 28,
        baseStats: {
          hp: 70,
          attackDamage: 10,
          attackSpeed: 0.5,
          armor: 1,
          movementSpeed: 1.0,
          visionRange: 9,
          attackRange: 7,
        },
        upgradeLevel: 0,
        maxUpgradeLevel: 3,
        prerequisites: [{ researchId: 'healing' }],
        abilities: [
          {
            id: 'heal',
            name: 'Heal',
            description: 'Restore health to an ally',
            cooldown: 5,
            manaCost: 15,
          },
          {
            id: 'group_heal',
            name: 'Group Heal',
            description: 'Heal multiple allies',
            cooldown: 25,
            manaCost: 50,
          },
        ],
      },
    ];

    units.forEach((unit) => {
      this.unitDefinitions.set(unit.type, unit);
    });
  }

  // Initialize unit upgrades
  private initializeUnitUpgrades() {
    const upgrades: UnitUpgrade[] = [
      // Warrior upgrades
      {
        id: 'warrior_upgrade_1',
        unitType: UnitType.WARRIOR,
        level: 1,
        name: 'Warrior Training I',
        description: 'Improve warrior combat capabilities',
        cost: { gold: 300, food: 150 },
        researchTime: 45,
        statBonuses: {
          hp: 20,
          attackDamage: 3,
          armor: 1,
        },
        prerequisiteResearch: ['melee_attack_1', 'armor_1'],
      },
      {
        id: 'warrior_upgrade_2',
        unitType: UnitType.WARRIOR,
        level: 2,
        name: 'Warrior Training II',
        description: 'Further improve warrior effectiveness',
        cost: { gold: 600, food: 300, stone: 100 },
        researchTime: 90,
        statBonuses: {
          hp: 30,
          attackDamage: 5,
          armor: 2,
        },
        prerequisiteResearch: ['melee_attack_2', 'armor_2'],
      },
      {
        id: 'warrior_upgrade_3',
        unitType: UnitType.WARRIOR,
        level: 3,
        name: 'Warrior Training III',
        description: 'Elite warrior training',
        cost: { gold: 1200, food: 600, stone: 200 },
        researchTime: 150,
        statBonuses: {
          hp: 50,
          attackDamage: 8,
          armor: 3,
          attackSpeed: 0.2,
        },
        prerequisiteResearch: ['melee_attack_3', 'armor_3'],
      },
      // Archer upgrades
      {
        id: 'archer_upgrade_1',
        unitType: UnitType.ARCHER,
        level: 1,
        name: 'Archer Training I',
        description: 'Improve archer accuracy and damage',
        cost: { gold: 250, food: 100, wood: 100 },
        researchTime: 40,
        statBonuses: {
          hp: 15,
          attackDamage: 3,
          attackRange: 1,
        },
        prerequisiteResearch: ['ranged_attack_1'],
      },
      {
        id: 'archer_upgrade_2',
        unitType: UnitType.ARCHER,
        level: 2,
        name: 'Archer Training II',
        description: 'Advanced archery techniques',
        cost: { gold: 500, food: 200, wood: 200 },
        researchTime: 80,
        statBonuses: {
          hp: 25,
          attackDamage: 5,
          attackRange: 2,
          attackSpeed: 0.1,
        },
        prerequisiteResearch: ['ranged_attack_2'],
      },
      // Cavalry upgrades
      {
        id: 'cavalry_upgrade_1',
        unitType: UnitType.CAVALRY,
        level: 1,
        name: 'Cavalry Training I',
        description: 'Improve cavalry speed and power',
        cost: { gold: 400, food: 250 },
        researchTime: 50,
        statBonuses: {
          hp: 25,
          attackDamage: 4,
          movementSpeed: 0.2,
        },
        prerequisiteResearch: ['cavalry_speed'],
      },
      // Hero upgrades
      {
        id: 'hero_upgrade_1',
        unitType: UnitType.HERO,
        level: 1,
        name: 'Hero Enhancement I',
        description: 'Enhance hero abilities',
        cost: { gold: 1000, food: 400, mana: 200 },
        researchTime: 120,
        statBonuses: {
          hp: 100,
          attackDamage: 10,
          armor: 2,
        },
        prerequisiteResearch: ['hero_unit'],
      },
      {
        id: 'hero_upgrade_2',
        unitType: UnitType.HERO,
        level: 2,
        name: 'Hero Enhancement II',
        description: 'Legendary hero enhancement',
        cost: { gold: 2000, food: 800, mana: 400 },
        researchTime: 180,
        statBonuses: {
          hp: 150,
          attackDamage: 15,
          armor: 3,
          attackSpeed: 0.3,
        },
        prerequisiteResearch: ['hero_unit', 'melee_attack_3'],
      },
      // Mage upgrades
      {
        id: 'mage_upgrade_1',
        unitType: UnitType.MAGE,
        level: 1,
        name: 'Mage Training I',
        description: 'Improve magical power',
        cost: { gold: 450, mana: 150 },
        researchTime: 60,
        statBonuses: {
          hp: 15,
          attackDamage: 8,
        },
        prerequisiteResearch: ['advanced_magic'],
      },
      {
        id: 'mage_upgrade_2',
        unitType: UnitType.MAGE,
        level: 2,
        name: 'Mage Training II',
        description: 'Arcane mastery enhancement',
        cost: { gold: 900, mana: 300 },
        researchTime: 120,
        statBonuses: {
          hp: 25,
          attackDamage: 15,
          attackRange: 2,
        },
        prerequisiteResearch: ['arcane_mastery'],
      },
    ];

    upgrades.forEach((upgrade) => {
      this.unitUpgrades.set(upgrade.id, upgrade);
    });
  }

  // Resource management
  getResources(): Resources {
    return { ...this.resources };
  }

  addResources(amount: Partial<Resources>): void {
    if (amount.gold) this.resources.gold += amount.gold;
    if (amount.wood) this.resources.wood += amount.wood;
    if (amount.stone) this.resources.stone += amount.stone;
    if (amount.food) this.resources.food += amount.food;
    if (amount.mana) this.resources.mana += amount.mana;
  }

  deductResources(amount: Partial<Resources>): boolean {
    if (!this.canAffordResources(amount)) {
      return false;
    }

    if (amount.gold) this.resources.gold -= amount.gold;
    if (amount.wood) this.resources.wood -= amount.wood;
    if (amount.stone) this.resources.stone -= amount.stone;
    if (amount.food) this.resources.food -= amount.food;
    if (amount.mana) this.resources.mana -= amount.mana;

    return true;
  }

  canAffordResources(cost: Partial<Resources>): boolean {
    if (cost.gold && this.resources.gold < cost.gold) return false;
    if (cost.wood && this.resources.wood < cost.wood) return false;
    if (cost.stone && this.resources.stone < cost.stone) return false;
    if (cost.food && this.resources.food < cost.food) return false;
    if (cost.mana && this.resources.mana < cost.mana) return false;
    return true;
  }

  // Building management
  upgradeBuilding(buildingInstanceId: string): boolean {
    const instance = this.buildings.get(buildingInstanceId);
    if (!instance) return false;

    const definition = this.buildingDefinitions.get(
      instance.buildingId as BuildingType
    );
    if (!definition) return false;

    if (instance.level >= definition.maxLevel) return false;
    if (instance.isUpgrading) return false;

    const cost = this.calculateUpgradeCost(definition, instance.level);
    if (!this.deductResources(cost)) return false;

    instance.isUpgrading = true;
    instance.upgradeProgress = 0;

    return true;
  }

  private calculateUpgradeCost(
    building: Building,
    currentLevel: number
  ): BuildingCost {
    const multiplier = 1 + currentLevel * 0.5; // 50% increase per level
    return {
      gold: Math.floor(building.upgradeCost.gold * multiplier),
      wood: building.upgradeCost.wood
        ? Math.floor(building.upgradeCost.wood * multiplier)
        : 0,
      stone: building.upgradeCost.stone
        ? Math.floor(building.upgradeCost.stone * multiplier)
        : 0,
      food: building.upgradeCost.food
        ? Math.floor(building.upgradeCost.food * multiplier)
        : 0,
    };
  }

  canUpgradeBuilding(buildingType: BuildingType): boolean {
    const definition = this.buildingDefinitions.get(buildingType);
    if (!definition) return false;

    // Check prerequisites
    return this.checkBuildingPrerequisites(definition.prerequisites);
  }

  private checkBuildingPrerequisites(
    prerequisites: BuildingPrerequisite[]
  ): boolean {
    return prerequisites.every((prereq) => {
      if (prereq.researchId) {
        return this.researchStore
          .getCompletedResearches()
          .has(prereq.researchId);
      }
      if (prereq.buildingType && prereq.level) {
        // Check if we have a building of this type at this level
        for (const instance of this.buildings.values()) {
          if (
            instance.buildingId === prereq.buildingType &&
            instance.level >= prereq.level
          ) {
            return true;
          }
        }
        return false;
      }
      return true;
    });
  }

  isBuildingUnlocked(buildingType: BuildingType): boolean {
    return this.unlockedBuildings.has(buildingType);
  }

  unlockBuilding(buildingType: BuildingType): void {
    this.unlockedBuildings.add(buildingType);
  }

  getUnlockedBuildings(): BuildingType[] {
    return Array.from(this.unlockedBuildings);
  }

  // Unit management
  isUnitUnlocked(unitType: UnitType): boolean {
    return this.unlockedUnits.has(unitType);
  }

  unlockUnit(unitType: UnitType): void {
    this.unlockedUnits.add(unitType);
  }

  getUnlockedUnits(): UnitType[] {
    return Array.from(this.unlockedUnits);
  }

  trainUnit(unitType: UnitType): boolean {
    const definition = this.unitDefinitions.get(unitType);
    if (!definition) return false;

    if (!this.isUnitUnlocked(unitType)) return false;

    if (!this.deductResources(definition.baseCost)) return false;

    // Training logic would go here
    return true;
  }

  getUnitUpgrades(unitType: UnitType): UnitUpgrade[] {
    return Array.from(this.unitUpgrades.values()).filter(
      (u) => u.unitType === unitType
    );
  }

  getBuildingDefinition(type: BuildingType): Building | undefined {
    return this.buildingDefinitions.get(type);
  }

  getUnitDefinition(type: UnitType): Unit | undefined {
    return this.unitDefinitions.get(type);
  }

  getAllBuildingDefinitions(): Building[] {
    return Array.from(this.buildingDefinitions.values());
  }

  getAllUnitDefinitions(): Unit[] {
    return Array.from(this.unitDefinitions.values());
  }

  // Update method for game loop
  update(deltaTime: number): void {
    // Update building upgrades
    for (const instance of this.buildings.values()) {
      if (instance.isUpgrading) {
        const definition = this.buildingDefinitions.get(
          instance.buildingId as BuildingType
        );
        if (definition) {
          instance.upgradeProgress += deltaTime / (definition.upgradeTime * 1000);
          if (instance.upgradeProgress >= 1) {
            instance.level++;
            instance.isUpgrading = false;
            instance.upgradeProgress = 0;
          }
        }
      }
    }

    // Update resource generation from buildings
    this.updateResourceGeneration(deltaTime);
  }

  private updateResourceGeneration(deltaTime: number): void {
    // Add passive resource generation from buildings
    for (const instance of this.buildings.values()) {
      const definition = this.buildingDefinitions.get(
        instance.buildingId as BuildingType
      );
      if (definition) {
        // Check for resource generation effects
        definition.effects.forEach((effect) => {
          if (effect.type === 'resource_generation') {
            this.resources.gold += effect.value * instance.level * (deltaTime / 1000);
          }
          if (effect.type === 'mana_generation') {
            this.resources.mana += effect.value * instance.level * (deltaTime / 1000);
          }
        });
      }
    }
  }
}
