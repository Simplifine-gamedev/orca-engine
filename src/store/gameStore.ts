// Game Store - Handles game state and configuration
// ORC-168: Early game pacing improvements

export interface Resources {
  food: number;
  wood: number;
  stone: number;
  gold: number;
}

export interface Unit {
  id: string;
  type: string;
  position: { x: number; y: number };
  health: number;
  speed: number;
}

export interface Building {
  id: string;
  type: string;
  position: { x: number; y: number };
  buildTime: number;
  health: number;
}

export interface GameState {
  resources: Resources;
  units: Unit[];
  buildings: Building[];
  gameTime: number;
}

// ORC-168 Fix #1: Increase starting resources (doubled from original values)
export const STARTING_RESOURCES: Resources = {
  food: 400,    // Increased from 200
  wood: 300,    // Increased from 150
  stone: 200,   // Increased from 100
  gold: 100,    // Increased from 50
};

// ORC-168 Fix #2: Speed up early resource gathering (increased by 50%)
export const RESOURCE_GATHERING_RATES = {
  food: 15,     // Increased from 10
  wood: 12,     // Increased from 8
  stone: 9,     // Increased from 6
  gold: 6,      // Increased from 4
};

// ORC-168 Fix #3: Scout unit for early exploration
export const UNIT_TYPES = {
  scout: {
    cost: { food: 50, wood: 0, stone: 0, gold: 0 },
    buildTime: 5,  // Very fast build time (5 seconds)
    speed: 8,      // Fast movement speed
    health: 75,
    attackDamage: 5,
    visionRange: 12,  // Extra large vision range for exploration
    description: "Fast exploration unit with large vision range",
  },
  worker: {
    cost: { food: 50, wood: 0, stone: 0, gold: 0 },
    buildTime: 10,  // Reduced from 15
    speed: 4,
    health: 100,
    attackDamage: 5,
    visionRange: 6,
    description: "Basic resource gatherer",
  },
  soldier: {
    cost: { food: 60, wood: 20, stone: 0, gold: 0 },
    buildTime: 12,  // Reduced from 20
    speed: 5,
    health: 150,
    attackDamage: 15,
    visionRange: 8,
    description: "Basic combat unit",
  },
  archer: {
    cost: { food: 50, wood: 30, stone: 0, gold: 10 },
    buildTime: 15,  // Reduced from 25
    speed: 4,
    health: 100,
    attackDamage: 20,
    visionRange: 10,
    description: "Ranged combat unit",
  },
};

// ORC-168 Fix #5: Reduce build times for first buildings
export const BUILDING_TYPES = {
  townCenter: {
    cost: { food: 0, wood: 0, stone: 0, gold: 0 },
    buildTime: 0,
    health: 2000,
    produces: ["worker", "scout"],
    description: "Main building, produces workers and scouts",
  },
  house: {
    cost: { food: 0, wood: 50, stone: 0, gold: 0 },
    buildTime: 15,  // Reduced from 30
    health: 500,
    populationBonus: 5,
    description: "Increases population cap",
  },
  barracks: {
    cost: { food: 0, wood: 100, stone: 50, gold: 0 },
    buildTime: 25,  // Reduced from 45
    health: 1000,
    produces: ["soldier", "archer"],
    description: "Produces military units",
  },
  farm: {
    cost: { food: 0, wood: 75, stone: 0, gold: 0 },
    buildTime: 20,  // Reduced from 35
    health: 500,
    resourceGeneration: { food: 8, wood: 0, stone: 0, gold: 0 },
    description: "Generates food over time",
  },
  lumberMill: {
    cost: { food: 0, wood: 100, stone: 0, gold: 0 },
    buildTime: 25,  // Reduced from 40
    health: 750,
    gatheringBonus: { food: 0, wood: 1.25, stone: 0, gold: 0 },
    description: "Increases wood gathering by 25%",
  },
  stoneMine: {
    cost: { food: 0, wood: 75, stone: 0, gold: 0 },
    buildTime: 30,  // Reduced from 50
    health: 1000,
    resourceGeneration: { food: 0, wood: 0, stone: 5, gold: 0 },
    description: "Generates stone over time",
  },
};

// ORC-168 Fix #4: Add early game objectives/quests
export interface Quest {
  id: string;
  title: string;
  description: string;
  objectives: {
    type: string;
    target: string | number;
    current: number;
    required: number;
  }[];
  rewards: Resources;
  completed: boolean;
}

export const EARLY_GAME_QUESTS: Quest[] = [
  {
    id: "tutorial_1",
    title: "First Steps",
    description: "Train your first scout to explore the map",
    objectives: [
      {
        type: "train_unit",
        target: "scout",
        current: 0,
        required: 1,
      },
    ],
    rewards: { food: 100, wood: 50, stone: 0, gold: 0 },
    completed: false,
  },
  {
    id: "tutorial_2",
    title: "Gather Resources",
    description: "Collect 500 food to sustain your civilization",
    objectives: [
      {
        type: "collect_resource",
        target: "food",
        current: 0,
        required: 500,
      },
    ],
    rewards: { food: 0, wood: 100, stone: 50, gold: 25 },
    completed: false,
  },
  {
    id: "tutorial_3",
    title: "Build Your Base",
    description: "Construct your first house to increase population capacity",
    objectives: [
      {
        type: "build_building",
        target: "house",
        current: 0,
        required: 1,
      },
    ],
    rewards: { food: 150, wood: 100, stone: 50, gold: 0 },
    completed: false,
  },
  {
    id: "tutorial_4",
    title: "Expand Economy",
    description: "Build a farm to generate food automatically",
    objectives: [
      {
        type: "build_building",
        target: "farm",
        current: 0,
        required: 1,
      },
    ],
    rewards: { food: 200, wood: 150, stone: 100, gold: 50 },
    completed: false,
  },
  {
    id: "tutorial_5",
    title: "Train Your Army",
    description: "Build a barracks and train 2 soldiers",
    objectives: [
      {
        type: "build_building",
        target: "barracks",
        current: 0,
        required: 1,
      },
      {
        type: "train_unit",
        target: "soldier",
        current: 0,
        required: 2,
      },
    ],
    rewards: { food: 300, wood: 200, stone: 150, gold: 100 },
    completed: false,
  },
];

// Game store class
class GameStore {
  private state: GameState;
  private quests: Quest[];

  constructor() {
    this.state = {
      resources: { ...STARTING_RESOURCES },
      units: [],
      buildings: [],
      gameTime: 0,
    };
    this.quests = [...EARLY_GAME_QUESTS];
    this.initializeStartingUnits();
  }

  private initializeStartingUnits(): void {
    // Start with town center
    this.state.buildings.push({
      id: "tc_1",
      type: "townCenter",
      position: { x: 50, y: 50 },
      buildTime: 0,
      health: 2000,
    });

    // Start with 3 workers for faster early game
    for (let i = 0; i < 3; i++) {
      this.state.units.push({
        id: `worker_${i}`,
        type: "worker",
        position: { x: 50 + i * 2, y: 50 + i * 2 },
        health: 100,
        speed: 4,
      });
    }
  }

  getState(): GameState {
    return { ...this.state };
  }

  getQuests(): Quest[] {
    return [...this.quests];
  }

  addResources(resources: Partial<Resources>): void {
    Object.keys(resources).forEach((key) => {
      const resourceKey = key as keyof Resources;
      this.state.resources[resourceKey] += resources[resourceKey] || 0;
    });
  }

  spendResources(resources: Partial<Resources>): boolean {
    // Check if we have enough resources
    for (const key of Object.keys(resources)) {
      const resourceKey = key as keyof Resources;
      if (this.state.resources[resourceKey] < (resources[resourceKey] || 0)) {
        return false;
      }
    }

    // Spend the resources
    Object.keys(resources).forEach((key) => {
      const resourceKey = key as keyof Resources;
      this.state.resources[resourceKey] -= resources[resourceKey] || 0;
    });

    return true;
  }

  updateQuestProgress(questId: string, objectiveIndex: number, value: number): void {
    const quest = this.quests.find((q) => q.id === questId);
    if (quest && !quest.completed) {
      quest.objectives[objectiveIndex].current = value;

      // Check if quest is completed
      const allObjectivesComplete = quest.objectives.every(
        (obj) => obj.current >= obj.required
      );

      if (allObjectivesComplete) {
        quest.completed = true;
        this.addResources(quest.rewards);
      }
    }
  }

  update(deltaTime: number): void {
    this.state.gameTime += deltaTime;
    
    // Auto-generate resources from buildings
    this.state.buildings.forEach((building) => {
      const buildingType = BUILDING_TYPES[building.type as keyof typeof BUILDING_TYPES];
      if (buildingType && 'resourceGeneration' in buildingType) {
        const generation = buildingType.resourceGeneration;
        if (generation) {
          Object.keys(generation).forEach((key) => {
            const resourceKey = key as keyof Resources;
            // Generate resources per second
            this.state.resources[resourceKey] += 
              (generation[resourceKey] || 0) * deltaTime;
          });
        }
      }
    });
  }
}

export default GameStore;
