export interface Unit {
  id: string;
  name: string;
  health: number;
  maxHealth: number;
  position: { x: number; y: number };
  garrisonedIn?: string; // Building ID if garrisoned
}

export interface Building {
  id: string;
  name: string;
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  garrisonedUnits: string[]; // Array of unit IDs
  maxGarrison: number;
}

export interface GameState {
  units: Record<string, Unit>;
  buildings: Record<string, Building>;
  selectedBuildingId: string | null;
}
