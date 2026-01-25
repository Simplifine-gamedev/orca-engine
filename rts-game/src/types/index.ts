// Core types for the RTS game

export interface Position {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  type: 'worker' | 'soldier' | 'builder';
  position: Position;
  health: number;
  maxHealth: number;
  isSelected: boolean;
  isIdle: boolean;
  currentTask?: string;
}

export interface Building {
  id: string;
  type: 'base' | 'barracks' | 'resource';
  position: Position;
  health: number;
  maxHealth: number;
}

export interface Resource {
  wood: number;
  gold: number;
  stone: number;
}

export interface GameState {
  units: Unit[];
  buildings: Building[];
  resources: Resource;
  selectedUnits: string[];
}
