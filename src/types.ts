export interface Vector2 {
  x: number;
  y: number;
}

export enum UnitTeam {
  FRIENDLY = 'friendly',
  ENEMY = 'enemy',
  NEUTRAL = 'neutral'
}

export interface Unit {
  id: string;
  name: string;
  team: UnitTeam;
  position: Vector2;
  health: number;
  maxHealth: number;
  size: number;
  color: string;
}

export interface SelectionBox {
  start: Vector2;
  end: Vector2;
}

export interface GameState {
  units: Unit[];
  selectedUnitIds: string[];
  selectionBox: SelectionBox | null;
  isDragging: boolean;
}
