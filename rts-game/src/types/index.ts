export interface Position {
  x: number
  y: number
}

export interface Unit {
  id: string
  position: Position
  targetPosition: Position | null
  isSelected: boolean
  color: string
  facing: number // Angle in radians
}

export type FormationType = 'line' | 'box' | 'wedge'
export type SpreadType = 'tight' | 'normal' | 'loose'

export interface FormationConfig {
  type: FormationType
  spread: SpreadType
  facing: number
  showIndividualPaths: boolean
}

export interface GameState {
  units: Unit[]
  selectedUnitIds: string[]
  isDraggingFormation: boolean
  formationDragStart: Position | null
  formationDragEnd: Position | null
  formationConfig: FormationConfig
}
