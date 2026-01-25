/**
 * Faction configuration for Orca RTS
 * Defines unit types, buildings, and faction-specific properties
 */

export interface UnitConfig {
  id: string;
  name: string;
  type: 'melee' | 'ranged' | 'cavalry' | 'siege';
  health: number;
  attack: number;
  attackRange: number;
  moveSpeed: number;
  attackSpeed: number; // attacks per second
  cost: {
    gold: number;
    wood?: number;
    stone?: number;
  };
  trainingTime: number; // seconds
  modelPath: string;
  animations: {
    idle: string;
    walk: string;
    attack: string;
    death: string;
  };
  projectile?: {
    modelPath: string;
    speed: number;
    arc: number; // projectile arc height (0-1)
  };
}

export interface BuildingConfig {
  id: string;
  name: string;
  type: 'military' | 'economic' | 'defensive';
  health: number;
  cost: {
    gold: number;
    wood: number;
    stone?: number;
  };
  buildTime: number;
  producesUnits?: string[]; // unit IDs
  modelPath: string;
}

export interface FactionConfig {
  id: string;
  name: string;
  description: string;
  units: UnitConfig[];
  buildings: BuildingConfig[];
}

// Define Archer unit
const archerUnit: UnitConfig = {
  id: 'archer',
  name: 'Archer',
  type: 'ranged',
  health: 50,
  attack: 8,
  attackRange: 7,
  moveSpeed: 3.5,
  attackSpeed: 0.8,
  cost: {
    gold: 40,
    wood: 25,
  },
  trainingTime: 20,
  modelPath: 'res://models/units/archer.glb',
  animations: {
    idle: 'idle',
    walk: 'walk',
    attack: 'shoot_bow',
    death: 'death',
  },
  projectile: {
    modelPath: 'res://models/projectiles/arrow.glb',
    speed: 15,
    arc: 0.3,
  },
};

// Define Crossbowman unit
const crossbowmanUnit: UnitConfig = {
  id: 'crossbowman',
  name: 'Crossbowman',
  type: 'ranged',
  health: 60,
  attack: 12,
  attackRange: 8,
  moveSpeed: 3.0,
  attackSpeed: 0.5,
  cost: {
    gold: 60,
    wood: 40,
  },
  trainingTime: 30,
  modelPath: 'res://models/units/crossbowman.glb',
  animations: {
    idle: 'idle',
    walk: 'walk',
    attack: 'shoot_crossbow',
    death: 'death',
  },
  projectile: {
    modelPath: 'res://models/projectiles/bolt.glb',
    speed: 20,
    arc: 0.15,
  },
};

// Define Archery Range building
const archeryRangeBuilding: BuildingConfig = {
  id: 'archery_range',
  name: 'Archery Range',
  type: 'military',
  health: 800,
  cost: {
    gold: 150,
    wood: 100,
  },
  buildTime: 45,
  producesUnits: ['archer', 'crossbowman'],
  modelPath: 'res://models/buildings/archery_range.glb',
};

// Default human faction with archer units
export const humanFaction: FactionConfig = {
  id: 'human',
  name: 'Human Kingdom',
  description: 'A versatile faction with strong ranged units and defensive capabilities',
  units: [
    archerUnit,
    crossbowmanUnit,
    // Other units would be added here
  ],
  buildings: [
    archeryRangeBuilding,
    // Other buildings would be added here
  ],
};

// Export all factions
export const allFactions: FactionConfig[] = [
  humanFaction,
  // Other factions would be added here
];

// Helper function to get unit by ID
export function getUnitById(unitId: string): UnitConfig | undefined {
  for (const faction of allFactions) {
    const unit = faction.units.find(u => u.id === unitId);
    if (unit) return unit;
  }
  return undefined;
}

// Helper function to get building by ID
export function getBuildingById(buildingId: string): BuildingConfig | undefined {
  for (const faction of allFactions) {
    const building = faction.buildings.find(b => b.id === buildingId);
    if (building) return building;
  }
  return undefined;
}
