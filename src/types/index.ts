// Core types for the RTS game

export interface Position {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  position: Position;
  team: 'friendly' | 'enemy';
  isMoving: boolean;
  path?: Position[];
}

export interface Wall {
  id: string;
  position: Position;
  type: 'wall' | 'gate';
  isOpen?: boolean; // Only for gates
  team: 'friendly' | 'neutral';
}

export interface Gate extends Wall {
  type: 'gate';
  isOpen: boolean;
  lastOpenedTime?: number;
  closeDelay: number; // milliseconds to wait before closing
  detectionRadius: number; // radius to detect units
}

export interface PathNode {
  position: Position;
  g: number; // cost from start
  h: number; // heuristic to goal
  f: number; // total cost
  parent?: PathNode;
}
