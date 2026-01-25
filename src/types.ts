export interface Position {
  x: number;
  y: number;
}

export type UnitType = 'friendly' | 'enemy' | 'neutral';

export interface Unit {
  id: string;
  type: UnitType;
  position: Position;
  health: number;
  maxHealth: number;
  name: string;
  isSelected?: boolean;
}

export type CursorMode = 'default' | 'attack-enemy' | 'attack-neutral' | 'friendly' | 'move';
