export interface ControlPointData {
  id: string;
  position: { x: number; y: number; z: number };
  ownerId: string | null;
  captureProgress: number;
  captureRadius: number;
  name?: string;
}

export interface Player {
  id: string;
  name: string;
  color: string;
  team: number;
}

export enum ControlPointStatus {
  NEUTRAL = 'neutral',
  CONTROLLED = 'controlled',
  ENEMY = 'enemy',
  ALLY = 'ally',
  OTHER = 'other'
}
