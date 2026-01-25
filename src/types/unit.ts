export type AnimationState = 'idle' | 'walking' | 'attacking' | 'dying' | 'spawning';

export interface Unit {
  id: string;
  position: { x: number; y: number };
  type: string;
  health: number;
  maxHealth: number;
  animationState: AnimationState;
  isSpawning: boolean;
}

export interface Building {
  id: string;
  position: { x: number; y: number };
  type: string;
  spawnPoint: { x: number; y: number };
}
