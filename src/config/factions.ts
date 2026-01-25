// Faction and unit configuration for Orca RTS
// This file defines the base faction configurations and unit types

export interface UnitConfig {
  id: string;
  name: string;
  type: 'scout' | 'warrior' | 'worker' | 'siege';
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
  buildTime: number; // in seconds
  availableFrom: string[]; // building types that can produce this unit
  description: string;
}

export interface FactionConfig {
  id: string;
  name: string;
  description: string;
  units: UnitConfig[];
  bonuses?: {
    description: string;
    effects: Record<string, number>;
  }[];
}

// Scout unit configuration - designed for early game exploration
export const scoutUnitConfig: UnitConfig = {
  id: 'scout',
  name: 'Scout',
  type: 'scout',
  cost: {
    food: 50,
    gold: 25,
  },
  stats: {
    health: 60,
    attack: 5, // Low attack - not meant for combat
    defense: 2,
    movementSpeed: 8.5, // Fast movement for exploration
    visionRange: 12, // Large vision range for scouting
  },
  buildTime: 15, // Quick to train
  availableFrom: ['town_center', 'stable'],
  description: 'Fast reconnaissance unit with excellent vision range. Ideal for early game map exploration and scouting enemy positions.',
};

// Base faction configuration with scout unit
export const baseFaction: FactionConfig = {
  id: 'base',
  name: 'Base Faction',
  description: 'Standard faction with balanced units',
  units: [
    scoutUnitConfig,
    // Other units can be added here
  ],
};

// Export all factions
export const factions: FactionConfig[] = [
  baseFaction,
];

// Utility function to get unit by id
export function getUnitConfig(unitId: string): UnitConfig | undefined {
  for (const faction of factions) {
    const unit = faction.units.find(u => u.id === unitId);
    if (unit) return unit;
  }
  return undefined;
}

// Utility function to get units available from a specific building
export function getUnitsFromBuilding(buildingType: string): UnitConfig[] {
  const availableUnits: UnitConfig[] = [];
  for (const faction of factions) {
    for (const unit of faction.units) {
      if (unit.availableFrom.includes(buildingType)) {
        availableUnits.push(unit);
      }
    }
  }
  return availableUnits;
}
