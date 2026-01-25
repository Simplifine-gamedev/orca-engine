// Unit management utilities
import { Unit, Position } from '../types';
import { wallStore } from '../store/wallStore';
import { pathfinder } from '../pathfinding/pathfinding';

export class UnitManager {
  private units: Map<string, Unit>;
  private movementInterval: number | null = null;

  constructor() {
    this.units = new Map();
  }

  // Create a new unit
  createUnit(id: string, position: Position, team: 'friendly' | 'enemy'): Unit {
    const unit: Unit = {
      id,
      position,
      team,
      isMoving: false,
    };
    this.units.set(id, unit);
    wallStore.updateUnit(unit);
    return unit;
  }

  // Move unit to target position
  moveUnit(unitId: string, target: Position): boolean {
    const unit = this.units.get(unitId);
    if (!unit) return false;

    const path = pathfinder.findPath(unit.position, target, unit.team);
    if (!path) return false;

    unit.path = path;
    unit.isMoving = true;
    this.units.set(unitId, unit);
    wallStore.updateUnit(unit);
    return true;
  }

  // Update unit positions along their paths
  private updateUnitMovement(): void {
    for (const unit of this.units.values()) {
      if (!unit.isMoving || !unit.path || unit.path.length === 0) {
        continue;
      }

      // Move to next position in path
      unit.position = unit.path.shift()!;

      // Check if reached destination
      if (unit.path.length === 0) {
        unit.isMoving = false;
        unit.path = undefined;
      }

      this.units.set(unit.id, unit);
      wallStore.updateUnit(unit);
    }
  }

  // Start automatic unit movement updates
  startMovementUpdates(intervalMs: number = 200): void {
    if (this.movementInterval !== null) return;

    this.movementInterval = window.setInterval(() => {
      this.updateUnitMovement();
    }, intervalMs);
  }

  // Stop automatic unit movement updates
  stopMovementUpdates(): void {
    if (this.movementInterval !== null) {
      clearInterval(this.movementInterval);
      this.movementInterval = null;
    }
  }

  // Remove a unit
  removeUnit(unitId: string): void {
    this.units.delete(unitId);
    wallStore.removeUnit(unitId);
  }

  // Get all units
  getUnits(): Unit[] {
    return Array.from(this.units.values());
  }

  // Get unit by id
  getUnit(unitId: string): Unit | undefined {
    return this.units.get(unitId);
  }

  // Reset all units
  reset(): void {
    this.stopMovementUpdates();
    this.units.clear();
  }
}

export const unitManager = new UnitManager();
export default unitManager;
