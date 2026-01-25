export interface Position {
  x: number
  y: number
}

export interface Tree {
  id: string
  position: Position
  wood: number
  maxWood: number
  regrowthRate: number
  isBeingChopped: boolean
  lastChopTime: number
}

export interface Worker {
  id: string
  position: Position
  targetTreeId: string | null
  carryingWood: number
  maxCarryCapacity: number
  chopSpeed: number
  moveSpeed: number
  state: 'idle' | 'moving_to_tree' | 'chopping' | 'returning' | 'depositing'
}

export interface Building {
  id: string
  type: 'lumber_camp' | 'town_hall'
  position: Position
  gatherBonus: number
}

export interface Resources {
  wood: number
  gold: number
  food: number
}

export interface GameState {
  resources: Resources
  population: number
  maxPopulation: number
}
