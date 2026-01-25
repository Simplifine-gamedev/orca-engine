export interface Position {
  x: number;
  y: number;
}

export interface Unit {
  id: string;
  position: Position;
  health: number;
  maxHealth: number;
  attack: number;
  attackSpeed: number;
  movementSpeed: number;
  team: 'player' | 'enemy';
  isSelected: boolean;
  target?: string;
  lastAttackTime: number;
}

export type DamageType = 'physical' | 'magic' | 'critical';

export interface DamageEvent {
  id: string;
  position: Position;
  amount: number;
  type: DamageType;
  timestamp: number;
}

export interface GameSettings {
  showDamageNumbers: boolean;
}
