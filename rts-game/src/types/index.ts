// Type definitions for the RTS game

export type DamageType = 'physical' | 'magical' | 'fire' | 'ice' | 'poison' | 'healing';

export interface DamageEvent {
  id: string;
  amount: number;
  type: DamageType;
  x: number;
  y: number;
  timestamp: number;
}

export interface Unit {
  id: string;
  name: string;
  health: number;
  maxHealth: number;
  attack: number;
  x: number;
  y: number;
  team: 'ally' | 'enemy';
}

export interface GameSettings {
  showDamageNumbers: boolean;
  soundEnabled: boolean;
  musicVolume: number;
}
