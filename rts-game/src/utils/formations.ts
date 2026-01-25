import { Position, FormationType, SpreadType } from '../types'

const SPREAD_DISTANCES = {
  tight: 30,
  normal: 50,
  loose: 80,
}

export function calculateFormationPositions(
  centerPosition: Position,
  unitCount: number,
  formationType: FormationType,
  spread: SpreadType,
  facing: number
): Position[] {
  const spreadDistance = SPREAD_DISTANCES[spread]

  switch (formationType) {
    case 'line':
      return calculateLineFormation(centerPosition, unitCount, spreadDistance, facing)
    case 'box':
      return calculateBoxFormation(centerPosition, unitCount, spreadDistance, facing)
    case 'wedge':
      return calculateWedgeFormation(centerPosition, unitCount, spreadDistance, facing)
    default:
      return []
  }
}

function calculateLineFormation(
  center: Position,
  count: number,
  spread: number,
  facing: number
): Position[] {
  const positions: Position[] = []
  const startOffset = -((count - 1) * spread) / 2
  
  // Line perpendicular to facing direction
  const perpAngle = facing + Math.PI / 2
  
  for (let i = 0; i < count; i++) {
    const offset = startOffset + i * spread
    positions.push({
      x: center.x + Math.cos(perpAngle) * offset,
      y: center.y + Math.sin(perpAngle) * offset,
    })
  }
  
  return positions
}

function calculateBoxFormation(
  center: Position,
  count: number,
  spread: number,
  facing: number
): Position[] {
  const positions: Position[] = []
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)
  
  const startX = -((cols - 1) * spread) / 2
  const startY = -((rows - 1) * spread) / 2
  
  // Rotate box based on facing direction
  const cosAngle = Math.cos(facing)
  const sinAngle = Math.sin(facing)
  
  let idx = 0
  for (let row = 0; row < rows && idx < count; row++) {
    for (let col = 0; col < cols && idx < count; col++) {
      const localX = startX + col * spread
      const localY = startY + row * spread
      
      // Rotate around center
      const rotatedX = localX * cosAngle - localY * sinAngle
      const rotatedY = localX * sinAngle + localY * cosAngle
      
      positions.push({
        x: center.x + rotatedX,
        y: center.y + rotatedY,
      })
      idx++
    }
  }
  
  return positions
}

function calculateWedgeFormation(
  center: Position,
  count: number,
  spread: number,
  facing: number
): Position[] {
  const positions: Position[] = []
  
  // Wedge formation: narrower at front, wider at back
  let currentRow = 0
  let unitsPlaced = 0
  
  while (unitsPlaced < count) {
    const unitsInRow = Math.min(currentRow + 1, count - unitsPlaced)
    const rowY = -currentRow * spread
    const startX = -((unitsInRow - 1) * spread) / 2
    
    for (let i = 0; i < unitsInRow; i++) {
      const localX = startX + i * spread
      const localY = rowY
      
      // Rotate based on facing
      const cosAngle = Math.cos(facing)
      const sinAngle = Math.sin(facing)
      const rotatedX = localX * cosAngle - localY * sinAngle
      const rotatedY = localX * sinAngle + localY * cosAngle
      
      positions.push({
        x: center.x + rotatedX,
        y: center.y + rotatedY,
      })
      
      unitsPlaced++
      if (unitsPlaced >= count) break
    }
    
    currentRow++
  }
  
  return positions
}

export function calculateCenterPosition(positions: Position[]): Position {
  if (positions.length === 0) return { x: 0, y: 0 }
  
  const sum = positions.reduce(
    (acc, pos) => ({
      x: acc.x + pos.x,
      y: acc.y + pos.y,
    }),
    { x: 0, y: 0 }
  )
  
  return {
    x: sum.x / positions.length,
    y: sum.y / positions.length,
  }
}

export function calculateAngle(from: Position, to: Position): number {
  return Math.atan2(to.y - from.y, to.x - from.x)
}
