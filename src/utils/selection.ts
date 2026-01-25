import { Vector2, Unit, SelectionBox } from '../types';

export function isPointInBox(point: Vector2, box: SelectionBox): boolean {
  const minX = Math.min(box.start.x, box.end.x);
  const maxX = Math.max(box.start.x, box.end.x);
  const minY = Math.min(box.start.y, box.end.y);
  const maxY = Math.max(box.start.y, box.end.y);

  return (
    point.x >= minX &&
    point.x <= maxX &&
    point.y >= minY &&
    point.y <= maxY
  );
}

export function getUnitsInBox(units: Unit[], box: SelectionBox): Unit[] {
  return units.filter(unit => isPointInBox(unit.position, box));
}

export function getSelectionColor(unit: Unit, isSelected: boolean): string {
  if (!isSelected) return unit.color;
  
  // Visual distinction for selected units based on team
  switch (unit.team) {
    case 'friendly':
      return '#00FF00'; // Green for friendly
    case 'enemy':
      return '#FF0000'; // Red for enemy
    case 'neutral':
      return '#FFFF00'; // Yellow for neutral
    default:
      return unit.color;
  }
}

export function getSelectionBoxColor(units: Unit[], selectedIds: string[]): string {
  const selectedUnits = units.filter(u => selectedIds.includes(u.id));
  
  if (selectedUnits.length === 0) return 'rgba(0, 255, 0, 0.3)';
  
  const hasEnemy = selectedUnits.some(u => u.team === 'enemy');
  const hasFriendly = selectedUnits.some(u => u.team === 'friendly');
  
  // Mixed selection
  if (hasEnemy && hasFriendly) {
    return 'rgba(255, 255, 0, 0.3)'; // Yellow for mixed
  }
  
  // Enemy only
  if (hasEnemy) {
    return 'rgba(255, 0, 0, 0.3)'; // Red for enemy
  }
  
  // Friendly only
  return 'rgba(0, 255, 0, 0.3)'; // Green for friendly
}
