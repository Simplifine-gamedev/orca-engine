// Factions configuration for Orca RTS
// Defines all factions, their units, buildings, and properties

export interface UnitData {
  name: string;
  type: string;
  health: number;
  damage: number;
  range: number;
  speed: number;
  attack_cooldown: number;
  armor_penetration?: number;
  cost_wood: number;
  cost_gold: number;
  train_time: number;
  trained_at: string;
  model: string;
  icon: string;
}

export interface BuildingData {
  name: string;
  type: string;
  health: number;
  cost_wood: number;
  cost_gold: number;
  build_time: number;
  trains: string[];
  model: string;
  icon: string;
}

export interface FactionData {
  id: number;
  name: string;
  color: string;
  units: Record<string, UnitData>;
  buildings: Record<string, BuildingData>;
}

export const FACTIONS: Record<string, FactionData> = {
  human: {
    id: 0,
    name: "Human Kingdom",
    color: "#3366CC",
    units: {
      archer: {
        name: "Archer",
        type: "ranged_infantry",
        health: 60,
        damage: 12,
        range: 15,
        speed: 3.5,
        attack_cooldown: 1.5,
        cost_wood: 50,
        cost_gold: 25,
        train_time: 30,
        trained_at: "archery_range",
        model: "res://rts_game/units/archer.tscn",
        icon: "res://rts_game/assets/icons/archer_icon.png"
      },
      crossbowman: {
        name: "Crossbowman",
        type: "ranged_infantry",
        health: 70,
        damage: 18,
        range: 18,
        speed: 3.0,
        attack_cooldown: 2.2,
        armor_penetration: 5,
        cost_wood: 60,
        cost_gold: 40,
        train_time: 45,
        trained_at: "archery_range",
        model: "res://rts_game/units/crossbowman.tscn",
        icon: "res://rts_game/assets/icons/crossbowman_icon.png"
      }
    },
    buildings: {
      archery_range: {
        name: "Archery Range",
        type: "military",
        health: 500,
        cost_wood: 150,
        cost_gold: 50,
        build_time: 60,
        trains: ["archer", "crossbowman"],
        model: "res://rts_game/buildings/archery_range.tscn",
        icon: "res://rts_game/assets/icons/archery_range_icon.png"
      }
    }
  },
  orc: {
    id: 1,
    name: "Orc Horde",
    color: "#CC3333",
    units: {
      archer: {
        name: "Orc Archer",
        type: "ranged_infantry",
        health: 65,
        damage: 13,
        range: 14,
        speed: 3.3,
        attack_cooldown: 1.6,
        cost_wood: 50,
        cost_gold: 25,
        train_time: 28,
        trained_at: "archery_range",
        model: "res://rts_game/units/archer.tscn",
        icon: "res://rts_game/assets/icons/orc_archer_icon.png"
      },
      crossbowman: {
        name: "Orc Crossbowman",
        type: "ranged_infantry",
        health: 75,
        damage: 20,
        range: 17,
        speed: 2.8,
        attack_cooldown: 2.3,
        armor_penetration: 6,
        cost_wood: 60,
        cost_gold: 40,
        train_time: 42,
        trained_at: "archery_range",
        model: "res://rts_game/units/crossbowman.tscn",
        icon: "res://rts_game/assets/icons/orc_crossbowman_icon.png"
      }
    },
    buildings: {
      archery_range: {
        name: "War Lodge",
        type: "military",
        health: 550,
        cost_wood: 140,
        cost_gold: 60,
        build_time: 55,
        trains: ["archer", "crossbowman"],
        model: "res://rts_game/buildings/archery_range.tscn",
        icon: "res://rts_game/assets/icons/war_lodge_icon.png"
      }
    }
  },
  elf: {
    id: 2,
    name: "Elven Alliance",
    color: "#33CC66",
    units: {
      archer: {
        name: "Elven Archer",
        type: "ranged_infantry",
        health: 55,
        damage: 14,
        range: 18,
        speed: 4.0,
        attack_cooldown: 1.3,
        cost_wood: 55,
        cost_gold: 30,
        train_time: 32,
        trained_at: "archery_range",
        model: "res://rts_game/units/archer.tscn",
        icon: "res://rts_game/assets/icons/elven_archer_icon.png"
      },
      crossbowman: {
        name: "Elven Marksman",
        type: "ranged_infantry",
        health: 60,
        damage: 16,
        range: 20,
        speed: 3.5,
        attack_cooldown: 2.0,
        armor_penetration: 4,
        cost_wood: 65,
        cost_gold: 45,
        train_time: 40,
        trained_at: "archery_range",
        model: "res://rts_game/units/crossbowman.tscn",
        icon: "res://rts_game/assets/icons/elven_marksman_icon.png"
      }
    },
    buildings: {
      archery_range: {
        name: "Hunter's Hall",
        type: "military",
        health: 450,
        cost_wood: 160,
        cost_gold: 40,
        build_time: 50,
        trains: ["archer", "crossbowman"],
        model: "res://rts_game/buildings/archery_range.tscn",
        icon: "res://rts_game/assets/icons/hunters_hall_icon.png"
      }
    }
  }
};

export function getFactionData(factionName: string): FactionData | null {
  return FACTIONS[factionName] || null;
}

export function getUnitData(factionName: string, unitName: string): UnitData | null {
  const faction = getFactionData(factionName);
  return faction?.units[unitName] || null;
}

export function getBuildingData(factionName: string, buildingName: string): BuildingData | null {
  const faction = getFactionData(factionName);
  return faction?.buildings[buildingName] || null;
}

export function getAllFactions(): string[] {
  return Object.keys(FACTIONS);
}

export function getFactionColor(factionName: string): string {
  return FACTIONS[factionName]?.color || "#FFFFFF";
}
