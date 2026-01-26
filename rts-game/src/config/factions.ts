// Faction configuration for the RTS game
export interface FactionCharacter {
  id: string;
  name: string;
  previewImage: string;
  model_url?: string;
  thumbnail_url?: string;
  cost: number;
  buildTime: number;
}

export interface FactionBuilding {
  id: string;
  name: string;
  model_url?: string;
  thumbnail_url?: string;
  availableUnits: string[];
}

export interface Faction {
  id: string;
  name: string;
  units: {
    [key: string]: FactionCharacter;
  };
  buildings: {
    [key: string]: FactionBuilding;
  };
}

export const factions: { [key: string]: Faction } = {
  human: {
    id: 'human',
    name: 'Humans',
    units: {
      footman: {
        id: 'footman',
        name: 'Footman',
        previewImage: '/assets/units/human/footman_preview.png',
        thumbnail_url: '/assets/units/human/footman_preview.png',
        cost: 100,
        buildTime: 5
      },
      archer: {
        id: 'archer',
        name: 'Archer',
        previewImage: '/assets/units/human/archer_preview.png',
        thumbnail_url: '/assets/units/human/archer_preview.png',
        cost: 150,
        buildTime: 6
      },
      knight: {
        id: 'knight',
        name: 'Knight',
        previewImage: '/assets/units/human/knight_preview.png',
        thumbnail_url: '/assets/units/human/knight_preview.png',
        cost: 300,
        buildTime: 10
      }
    },
    buildings: {
      barracks: {
        id: 'barracks',
        name: 'Barracks',
        thumbnail_url: '/assets/buildings/human/barracks_thumbnail.png',
        availableUnits: ['footman', 'archer', 'knight']
      }
    }
  },
  dwarf: {
    id: 'dwarf',
    name: 'Dwarves',
    units: {
      warrior: {
        id: 'warrior',
        name: 'Dwarf Warrior',
        previewImage: '/assets/units/dwarf/warrior_preview.png',
        thumbnail_url: '/assets/units/dwarf/warrior_preview.png',
        cost: 120,
        buildTime: 5
      },
      rifleman: {
        id: 'rifleman',
        name: 'Rifleman',
        previewImage: '/assets/units/dwarf/rifleman_preview.png',
        thumbnail_url: '/assets/units/dwarf/rifleman_preview.png',
        cost: 180,
        buildTime: 7
      },
      hammerer: {
        id: 'hammerer',
        name: 'Hammerer',
        previewImage: '/assets/units/dwarf/hammerer_preview.png',
        thumbnail_url: '/assets/units/dwarf/hammerer_preview.png',
        cost: 350,
        buildTime: 12
      }
    },
    buildings: {
      barracks: {
        id: 'barracks',
        name: 'Barracks',
        thumbnail_url: '/assets/buildings/dwarf/barracks_thumbnail.png',
        availableUnits: ['warrior', 'rifleman', 'hammerer']
      }
    }
  },
  undead: {
    id: 'undead',
    name: 'Undead',
    units: {
      worker: {
        id: 'worker',
        name: 'Undead Worker',
        previewImage: '/assets/units/undead/worker_thumbnail.png',
        model_url: 'https://example.com/models/undead/worker.glb',
        thumbnail_url: '/assets/units/undead/worker_thumbnail.png',
        cost: 50,
        buildTime: 3
      },
      soldier_light: {
        id: 'soldier_light',
        name: 'Skeleton Warrior',
        previewImage: '/assets/units/undead/soldier_light_thumbnail.png',
        model_url: 'https://example.com/models/undead/soldier_light.glb',
        thumbnail_url: '/assets/units/undead/soldier_light_thumbnail.png',
        cost: 100,
        buildTime: 5
      },
      soldier_medium: {
        id: 'soldier_medium',
        name: 'Zombie Soldier',
        previewImage: '/assets/units/undead/soldier_medium_thumbnail.png',
        model_url: 'https://example.com/models/undead/soldier_medium.glb',
        thumbnail_url: '/assets/units/undead/soldier_medium_thumbnail.png',
        cost: 150,
        buildTime: 7
      },
      soldier_heavy: {
        id: 'soldier_heavy',
        name: 'Death Knight',
        previewImage: '/assets/units/undead/soldier_heavy_thumbnail.png',
        model_url: 'https://example.com/models/undead/soldier_heavy.glb',
        thumbnail_url: '/assets/units/undead/soldier_heavy_thumbnail.png',
        cost: 300,
        buildTime: 10
      }
    },
    buildings: {
      city_center: {
        id: 'city_center',
        name: 'Necropolis',
        model_url: 'https://example.com/models/undead/city_center.glb',
        thumbnail_url: '/assets/buildings/undead/city_center_thumbnail.png',
        availableUnits: ['worker']
      },
      barracks: {
        id: 'barracks',
        name: 'Crypt',
        model_url: 'https://example.com/models/undead/barracks.glb',
        thumbnail_url: '/assets/buildings/undead/barracks_thumbnail.png',
        availableUnits: ['soldier_light', 'soldier_medium', 'soldier_heavy']
      },
      farm: {
        id: 'farm',
        name: 'Graveyard',
        model_url: 'https://example.com/models/undead/farm.glb',
        thumbnail_url: '/assets/buildings/undead/farm_thumbnail.png',
        availableUnits: []
      },
      bank: {
        id: 'bank',
        name: 'Haunted Treasury',
        model_url: 'https://example.com/models/undead/bank.glb',
        thumbnail_url: '/assets/buildings/undead/bank_thumbnail.png',
        availableUnits: []
      },
      mill: {
        id: 'mill',
        name: 'Bone Mill',
        model_url: 'https://example.com/models/undead/mill.glb',
        thumbnail_url: '/assets/buildings/undead/mill_thumbnail.png',
        availableUnits: []
      },
      warehouse: {
        id: 'warehouse',
        name: 'Tomb Storage',
        model_url: 'https://example.com/models/undead/warehouse.glb',
        thumbnail_url: '/assets/buildings/undead/warehouse_thumbnail.png',
        availableUnits: []
      },
      tower: {
        id: 'tower',
        name: 'Spirit Tower',
        model_url: 'https://example.com/models/undead/tower.glb',
        thumbnail_url: '/assets/buildings/undead/tower_thumbnail.png',
        availableUnits: []
      }
    }
  }
};

export function getFaction(factionId: string): Faction | undefined {
  return factions[factionId];
}

export function getUnitPreview(factionId: string, unitId: string): string | undefined {
  const faction = getFaction(factionId);
  if (!faction) return undefined;
  
  const unit = faction.units[unitId];
  // Prefer thumbnail_url, fallback to previewImage
  return unit?.thumbnail_url || unit?.previewImage;
}

export function getBuildingThumbnail(factionId: string, buildingId: string): string | undefined {
  const faction = getFaction(factionId);
  if (!faction) return undefined;
  
  const building = faction.buildings[buildingId];
  return building?.thumbnail_url;
}

export function getBuildingUnits(factionId: string, buildingId: string): string[] {
  const faction = getFaction(factionId);
  if (!faction) return [];
  
  const building = faction.buildings[buildingId];
  return building?.availableUnits || [];
}
