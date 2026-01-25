export interface Position {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  position: Position;
  targetPosition: Position | null;
  health: number;
  maxHealth: number;
  team: 'player' | 'enemy';
  isSelected: boolean;
  type: 'soldier' | 'tank' | 'scout';
  speed: number;
}

export interface ControlGroup {
  [key: number]: string[]; // key is 1-9, value is array of unit IDs
}

export type ActionType = 'move' | 'attack' | null;
