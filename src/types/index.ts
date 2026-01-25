export interface Unit {
  id: string;
  factionId: string;
  type: string;
  health: number;
  maxHealth: number;
}

export interface Faction {
  id: string;
  name: string;
  color: string;
  population: number;
  maxPopulation: number;
  units: Unit[];
}

export interface GameState {
  factions: Record<string, Faction>;
  playerFactionId: string | null;
  worldPopulation: number;
}

export interface Resources {
  gold: number;
  wood: number;
  food: number;
}
