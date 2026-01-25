// Faction and Unit Configuration
export interface UnitConfig {
  id: string;
  name: string;
  cost: {
    gold?: number;
    wood?: number;
    food?: number;
  };
  stats: {
    health: number;
    attack: number;
    defense: number;
    movementSpeed: number;
    visionRange: number;
  };
  buildTime: number;
  availableFrom: string[]; // Buildings that can produce this unit
  description: string;
}

export interface FactionConfig {
  id: string;
  name: string;
  units: UnitConfig[];
  buildings: string[];
}

// Scout Unit Configuration
export const scoutUnit: UnitConfig = {
  id: "scout",
  name: "Scout",
  cost: {
    gold: 50,
    food: 25,
  },
  stats: {
    health: 60,
    attack: 5,
    defense: 2,
    movementSpeed: 8.0, // Fast movement speed
    visionRange: 15, // Large vision range
  },
  buildTime: 15,
  availableFrom: ["town_center", "stable"],
  description: "A fast, lightly armored unit ideal for early game exploration. High vision range and speed but minimal combat ability.",
};

// Default faction with scout unit
export const defaultFaction: FactionConfig = {
  id: "default",
  name: "Default Faction",
  units: [
    scoutUnit,
    {
      id: "warrior",
      name: "Warrior",
      cost: {
        gold: 100,
        food: 50,
      },
      stats: {
        health: 150,
        attack: 15,
        defense: 10,
        movementSpeed: 3.5,
        visionRange: 8,
      },
      buildTime: 30,
      availableFrom: ["barracks"],
      description: "Basic melee combat unit with balanced stats.",
    },
    {
      id: "archer",
      name: "Archer",
      cost: {
        gold: 80,
        wood: 40,
      },
      stats: {
        health: 80,
        attack: 12,
        defense: 5,
        movementSpeed: 4.0,
        visionRange: 10,
      },
      buildTime: 25,
      availableFrom: ["archery_range"],
      description: "Ranged combat unit effective against light armor.",
    },
  ],
  buildings: ["town_center", "barracks", "archery_range", "stable"],
};

// Export all factions
export const factions: FactionConfig[] = [defaultFaction];

// Helper function to get unit by ID
export function getUnitById(unitId: string): UnitConfig | undefined {
  for (const faction of factions) {
    const unit = faction.units.find((u) => u.id === unitId);
    if (unit) return unit;
  }
  return undefined;
}

// Helper function to get units available from a building
export function getUnitsFromBuilding(buildingId: string): UnitConfig[] {
  const availableUnits: UnitConfig[] = [];
  for (const faction of factions) {
    for (const unit of faction.units) {
      if (unit.availableFrom.includes(buildingId)) {
        availableUnits.push(unit);
      }
    }
  }
  return availableUnits;
}
