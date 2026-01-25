// Game types for RTS

export interface Position {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  position: Position;
  type: UnitType;
  isSelected: boolean;
  isIdle: boolean;
  currentTask?: Task;
}

export enum UnitType {
  WORKER = 'worker',
  SOLDIER = 'soldier',
  BUILDING = 'building',
}

export interface Task {
  type: TaskType;
  target?: Position;
  startTime: number;
}

export enum TaskType {
  IDLE = 'idle',
  GATHER = 'gather',
  BUILD = 'build',
  ATTACK = 'attack',
  MOVE = 'move',
}

export interface GameState {
  units: Unit[];
  selectedUnits: string[];
  resources: Resources;
}

export interface Resources {
  wood: number;
  food: number;
  gold: number;
  stone: number;
}
