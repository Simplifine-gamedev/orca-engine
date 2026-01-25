// Type definitions for the RTS game

export interface Vector2 {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  position: Vector2;
  targetPosition: Vector2 | null;
  selected: boolean;
  path: Vector2[];
  facingAngle: number;
}

export type FormationType = 'none' | 'line' | 'box' | 'wedge';

export type SpreadType = 'tight' | 'normal' | 'loose';

export interface FormationSettings {
  type: FormationType;
  spread: SpreadType;
  facingAngle: number;
  showIndividualPaths: boolean;
  showGroupPath: boolean;
}

export interface GameState {
  units: Unit[];
  selectedUnits: string[];
  formationSettings: FormationSettings;
  isDraggingFormation: boolean;
  formationDragStart: Vector2 | null;
  formationDragEnd: Vector2 | null;
}
