// Faction configuration for the RTS game
export interface Faction {
  id: string;
  name: string;
  units: {
    [key: string]: {
      id: string;
      name: string;
      previewImage: string;
      cost: number;
      buildTime: number;
    };
  };
  buildings: {
    [key: string]: {
      id: string;
      name: string;
      availableUnits: string[];
    };
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
        cost: 100,
        buildTime: 5
      },
      archer: {
        id: 'archer',
        name: 'Archer',
        previewImage: '/assets/units/human/archer_preview.png',
        cost: 150,
        buildTime: 6
      },
      knight: {
        id: 'knight',
        name: 'Knight',
        previewImage: '/assets/units/human/knight_preview.png',
        cost: 300,
        buildTime: 10
      }
    },
    buildings: {
      barracks: {
        id: 'barracks',
        name: 'Barracks',
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
        cost: 120,
        buildTime: 5
      },
      rifleman: {
        id: 'rifleman',
        name: 'Rifleman',
        previewImage: '/assets/units/dwarf/rifleman_preview.png',
        cost: 180,
        buildTime: 7
      },
      hammerer: {
        id: 'hammerer',
        name: 'Hammerer',
        previewImage: '/assets/units/dwarf/hammerer_preview.png',
        cost: 350,
        buildTime: 12
      }
    },
    buildings: {
      barracks: {
        id: 'barracks',
        name: 'Barracks',
        availableUnits: ['warrior', 'rifleman', 'hammerer']
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
  return unit?.previewImage;
}

export function getBuildingUnits(factionId: string, buildingId: string): string[] {
  const faction = getFaction(factionId);
  if (!faction) return [];
  
  const building = faction.buildings[buildingId];
  return building?.availableUnits || [];
}
