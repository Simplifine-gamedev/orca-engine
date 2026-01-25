export interface Position {
  x: number;
  y: number;
}

export interface WallSegment {
  id: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  cost: number;
  isValid: boolean;
}

export interface TerrainTile {
  x: number;
  y: number;
  type: 'grass' | 'stone' | 'water' | 'obstacle';
  buildable: boolean;
}

export interface BuildingCosts {
  wall: number;
  tower: number;
  gate: number;
}

export interface GameResources {
  wood: number;
  stone: number;
  gold: number;
}
