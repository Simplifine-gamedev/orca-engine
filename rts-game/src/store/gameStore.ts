/**
 * Game Store - Manages game state and resources for Orca RTS
 * 
 * PACING IMPROVEMENTS (ORC-145):
 * - Increased starting resources to speed up early game
 * - Faster resource gathering rates
 * - Reduced build times for first buildings
 */

export interface Resources {
  gold: number;
  wood: number;
  food: number;
  stone: number;
}

export interface Unit {
  id: string;
  type: 'villager' | 'scout' | 'warrior' | 'archer';
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  speed: number;
  gatherRate?: number;
}

export interface Building {
  id: string;
  type: 'town_center' | 'barracks' | 'farm' | 'lumber_mill' | 'mining_camp';
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  buildTime: number;
  isBuilding: boolean;
  buildProgress: number;
}

export interface GameState {
  resources: Resources;
  units: Unit[];
  buildings: Building[];
  population: number;
  maxPopulation: number;
  gameTime: number;
}

/**
 * Starting resources - INCREASED for ORC-145
 * Previous values: gold: 100, wood: 100, food: 100, stone: 50
 * New values: gold: 200, wood: 200, food: 150, stone: 100
 */
export const STARTING_RESOURCES: Resources = {
  gold: 200,    // +100% increase
  wood: 200,    // +100% increase
  food: 150,    // +50% increase
  stone: 100,   // +100% increase
};

/**
 * Resource gathering rates - INCREASED for ORC-145
 * Previous values: 0.5 per second
 * New values: 1.0 per second (2x faster)
 */
export const GATHERING_RATES = {
  villager: {
    gold: 1.0,   // +100% increase
    wood: 1.0,   // +100% increase
    food: 1.2,   // +140% increase (food was specifically mentioned as slow)
    stone: 0.8,  // +60% increase
  },
  scout: {
    gold: 0.5,
    wood: 0.5,
    food: 0.8,
    stone: 0.5,
  }
};

/**
 * Build times (in seconds) - REDUCED for early buildings (ORC-145)
 * Previous values: 30+ seconds for first buildings
 * New values: 15-20 seconds for early game structures
 */
export const BUILD_TIMES = {
  town_center: 60,    // Keep high for balance
  barracks: 15,       // Reduced from 30s
  farm: 10,           // Reduced from 20s
  lumber_mill: 12,    // Reduced from 25s
  mining_camp: 12,    // Reduced from 25s
  house: 8,           // Reduced from 15s
};

/**
 * Unit costs - Adjusted for better early game (ORC-145)
 */
export const UNIT_COSTS = {
  villager: { gold: 50, food: 25 },
  scout: { gold: 40, food: 15 },      // NEW: Scout unit for early exploration
  warrior: { gold: 60, food: 40 },
  archer: { gold: 45, food: 35, wood: 20 },
};

/**
 * Unit stats
 */
export const UNIT_STATS = {
  villager: {
    health: 50,
    speed: 1.2,
    buildTime: 20,
  },
  scout: {
    health: 60,
    speed: 2.5,        // Fast for exploration
    buildTime: 15,     // Quick to build
    visionRadius: 150, // Large vision radius
  },
  warrior: {
    health: 100,
    speed: 1.5,
    buildTime: 30,
  },
  archer: {
    health: 70,
    speed: 1.3,
    buildTime: 25,
  },
};

/**
 * Starting units - INCREASED for ORC-145
 * Previous: 3 villagers
 * New: 4 villagers + 1 scout
 */
export const STARTING_UNITS = {
  villagers: 4,  // +1 villager
  scouts: 1,     // NEW: Add scout for early exploration
};

// Game Store Class
export class GameStore {
  private state: GameState;
  private listeners: Set<(state: GameState) => void>;

  constructor() {
    this.state = this.getInitialState();
    this.listeners = new Set();
  }

  getInitialState(): GameState {
    return {
      resources: { ...STARTING_RESOURCES },
      units: this.createStartingUnits(),
      buildings: this.createStartingBuildings(),
      population: STARTING_UNITS.villagers + STARTING_UNITS.scouts,
      maxPopulation: 10,
      gameTime: 0,
    };
  }

  createStartingUnits(): Unit[] {
    const units: Unit[] = [];
    
    // Create starting villagers
    for (let i = 0; i < STARTING_UNITS.villagers; i++) {
      units.push({
        id: `villager-${i}`,
        type: 'villager',
        position: { x: 100 + i * 20, y: 100 },
        health: UNIT_STATS.villager.health,
        maxHealth: UNIT_STATS.villager.health,
        speed: UNIT_STATS.villager.speed,
        gatherRate: GATHERING_RATES.villager.gold,
      });
    }
    
    // Create starting scout (NEW for ORC-145)
    for (let i = 0; i < STARTING_UNITS.scouts; i++) {
      units.push({
        id: `scout-${i}`,
        type: 'scout',
        position: { x: 150, y: 150 },
        health: UNIT_STATS.scout.health,
        maxHealth: UNIT_STATS.scout.health,
        speed: UNIT_STATS.scout.speed,
        gatherRate: GATHERING_RATES.scout.gold,
      });
    }
    
    return units;
  }

  createStartingBuildings(): Building[] {
    return [{
      id: 'town-center-0',
      type: 'town_center',
      position: { x: 200, y: 200 },
      health: 1000,
      maxHealth: 1000,
      buildTime: 0,
      isBuilding: false,
      buildProgress: 100,
    }];
  }

  getState(): GameState {
    return { ...this.state };
  }

  subscribe(listener: (state: GameState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach(listener => listener(this.getState()));
  }

  // Update resources
  updateResources(resources: Partial<Resources>) {
    this.state.resources = {
      ...this.state.resources,
      ...resources,
    };
    this.notify();
  }

  // Add resource (with gathering rate)
  gatherResource(type: keyof Resources, amount: number) {
    this.state.resources[type] = Math.max(0, this.state.resources[type] + amount);
    this.notify();
  }

  // Spend resources
  spendResources(cost: Partial<Resources>): boolean {
    // Check if player has enough resources
    for (const [resource, amount] of Object.entries(cost)) {
      if (this.state.resources[resource as keyof Resources] < (amount || 0)) {
        return false;
      }
    }
    
    // Deduct resources
    for (const [resource, amount] of Object.entries(cost)) {
      this.state.resources[resource as keyof Resources] -= (amount || 0);
    }
    
    this.notify();
    return true;
  }

  // Add unit
  addUnit(unit: Unit) {
    this.state.units.push(unit);
    this.state.population++;
    this.notify();
  }

  // Add building
  addBuilding(building: Building) {
    this.state.buildings.push(building);
    this.notify();
  }

  // Update game time
  tick(deltaTime: number) {
    this.state.gameTime += deltaTime;
    
    // Update building progress
    this.state.buildings.forEach(building => {
      if (building.isBuilding) {
        building.buildProgress += (deltaTime / building.buildTime) * 100;
        if (building.buildProgress >= 100) {
          building.isBuilding = false;
          building.buildProgress = 100;
        }
      }
    });
    
    this.notify();
  }

  // Reset game
  reset() {
    this.state = this.getInitialState();
    this.notify();
  }
}

// Singleton instance
export const gameStore = new GameStore();
export default gameStore;
