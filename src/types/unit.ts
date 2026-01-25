export interface Position {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  position: Position;
  health: number;
  maxHealth: number;
  type: 'warrior' | 'archer' | 'mage';
  team: 'player' | 'enemy';
  isMoving: boolean;
  targetPosition: Position | null;
  speed: number;
}

export interface SelectionBox {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

export type ControlGroups = {
  [key: number]: string[]; // key: 1-9, value: array of unit IDs
};
