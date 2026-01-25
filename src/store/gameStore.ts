// Game store with movement logic and formation controls

import { GameState, Unit, Vector2, FormationType, SpreadType } from '../types';

const SPREAD_MULTIPLIERS = {
  tight: 0.5,
  normal: 1.0,
  loose: 2.0,
};

const BASE_UNIT_SPACING = 60;

class GameStore {
  private state: GameState;
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.state = {
      units: [],
      selectedUnits: [],
      formationSettings: {
        type: 'none',
        spread: 'normal',
        facingAngle: 0,
        showIndividualPaths: true,
        showGroupPath: false,
      },
      isDraggingFormation: false,
      formationDragStart: null,
      formationDragEnd: null,
    };
    this.initializeUnits();
  }

  private initializeUnits() {
    // Create some test units
    const units: Unit[] = [];
    for (let i = 0; i < 12; i++) {
      const col = i % 4;
      const row = Math.floor(i / 4);
      units.push({
        id: `unit-${i}`,
        position: { x: 150 + col * 80, y: 150 + row * 80 },
        targetPosition: null,
        selected: false,
        path: [],
        facingAngle: 0,
      });
    }
    this.state.units = units;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach(listener => listener());
  }

  getState(): GameState {
    return this.state;
  }

  /**
   * Select a unit by ID. Can add to existing selection or replace it.
   * @param unitId - The ID of the unit to select
   * @param addToSelection - If true, adds to current selection; if false, replaces selection
   */
  selectUnit(unitId: string, addToSelection: boolean = false) {
    if (!addToSelection) {
      this.state.selectedUnits = [unitId];
      this.state.units.forEach(u => u.selected = u.id === unitId);
    } else {
      const index = this.state.selectedUnits.indexOf(unitId);
      if (index >= 0) {
        this.state.selectedUnits.splice(index, 1);
        const unit = this.state.units.find(u => u.id === unitId);
        if (unit) unit.selected = false;
      } else {
        this.state.selectedUnits.push(unitId);
        const unit = this.state.units.find(u => u.id === unitId);
        if (unit) unit.selected = true;
      }
    }
    this.notify();
  }

  /**
   * Select all units within a rectangular area (box selection).
   * @param start - One corner of the selection box
   * @param end - Opposite corner of the selection box
   */
  selectUnitsInArea(start: Vector2, end: Vector2) {
    const minX = Math.min(start.x, end.x);
    const maxX = Math.max(start.x, end.x);
    const minY = Math.min(start.y, end.y);
    const maxY = Math.max(start.y, end.y);

    this.state.selectedUnits = [];
    this.state.units.forEach(unit => {
      const inArea = unit.position.x >= minX && unit.position.x <= maxX &&
                     unit.position.y >= minY && unit.position.y <= maxY;
      unit.selected = inArea;
      if (inArea) {
        this.state.selectedUnits.push(unit.id);
      }
    });
    this.notify();
  }

  deselectAll() {
    this.state.selectedUnits = [];
    this.state.units.forEach(u => u.selected = false);
    this.notify();
  }

  setFormationType(type: FormationType) {
    this.state.formationSettings.type = type;
    this.notify();
  }

  setSpread(spread: SpreadType) {
    this.state.formationSettings.spread = spread;
    this.notify();
  }

  toggleIndividualPaths() {
    this.state.formationSettings.showIndividualPaths = !this.state.formationSettings.showIndividualPaths;
    this.notify();
  }

  toggleGroupPath() {
    this.state.formationSettings.showGroupPath = !this.state.formationSettings.showGroupPath;
    this.notify();
  }

  /**
   * Start dragging to set formation facing direction (Total War style).
   * @param position - The starting position (target location)
   */
  startFormationDrag(position: Vector2) {
    if (this.state.selectedUnits.length > 0) {
      this.state.isDraggingFormation = true;
      this.state.formationDragStart = position;
      this.state.formationDragEnd = position;
      this.notify();
    }
  }

  /**
   * Update formation drag to show direction preview.
   * @param position - Current mouse position during drag
   */
  updateFormationDrag(position: Vector2) {
    if (this.state.isDraggingFormation) {
      this.state.formationDragEnd = position;
      this.notify();
    }
  }

  /**
   * Complete formation drag and move units with the set facing direction.
   * Calculates facing angle from drag vector and executes move command.
   */
  endFormationDrag() {
    if (this.state.isDraggingFormation && this.state.formationDragStart && this.state.formationDragEnd) {
      const dx = this.state.formationDragEnd.x - this.state.formationDragStart.x;
      const dy = this.state.formationDragEnd.y - this.state.formationDragStart.y;
      const angle = Math.atan2(dy, dx);
      this.state.formationSettings.facingAngle = angle;
      this.moveSelectedUnits(this.state.formationDragStart);
    }
    this.state.isDraggingFormation = false;
    this.state.formationDragStart = null;
    this.state.formationDragEnd = null;
    this.notify();
  }

  /**
   * Move selected units to target position with current formation settings.
   * Calculates formation positions and sets unit targets.
   * @param targetCenter - The center point of the target formation
   */
  moveSelectedUnits(targetCenter: Vector2) {
    const selectedUnits = this.state.units.filter(u => u.selected);
    if (selectedUnits.length === 0) return;

    const positions = this.calculateFormationPositions(targetCenter, selectedUnits.length);
    
    selectedUnits.forEach((unit, index) => {
      unit.targetPosition = positions[index];
      unit.facingAngle = this.state.formationSettings.facingAngle;
      // Simple straight-line path for now
      unit.path = [unit.position, positions[index]];
    });

    this.notify();
  }

  private calculateFormationPositions(center: Vector2, count: number): Vector2[] {
    const { type, spread, facingAngle } = this.state.formationSettings;
    const spacing = BASE_UNIT_SPACING * SPREAD_MULTIPLIERS[spread];
    const positions: Vector2[] = [];

    switch (type) {
      case 'line':
        return this.calculateLineFormation(center, count, spacing, facingAngle);
      case 'box':
        return this.calculateBoxFormation(center, count, spacing, facingAngle);
      case 'wedge':
        return this.calculateWedgeFormation(center, count, spacing, facingAngle);
      default:
        // No formation - simple grid
        return this.calculateGridFormation(center, count, spacing);
    }
  }

  /**
   * Calculate positions for line formation.
   * Units are arranged in a single line perpendicular to facing direction.
   * @param center - Center point of the formation
   * @param count - Number of units
   * @param spacing - Distance between units
   * @param angle - Facing direction in radians
   */
  private calculateLineFormation(center: Vector2, count: number, spacing: number, angle: number): Vector2[] {
    const positions: Vector2[] = [];
    const perpAngle = angle + Math.PI / 2; // Perpendicular to facing direction
    
    for (let i = 0; i < count; i++) {
      const offset = (i - (count - 1) / 2) * spacing;
      positions.push({
        x: center.x + Math.cos(perpAngle) * offset,
        y: center.y + Math.sin(perpAngle) * offset,
      });
    }
    return positions;
  }

  /**
   * Calculate positions for box formation.
   * Units are arranged in a rectangular grid, rotated by facing angle.
   * @param center - Center point of the formation
   * @param count - Number of units
   * @param spacing - Distance between units
   * @param angle - Facing direction in radians
   */
  private calculateBoxFormation(center: Vector2, count: number, spacing: number, angle: number): Vector2[] {
    const positions: Vector2[] = [];
    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    
    for (let i = 0; i < count; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const localX = (col - (cols - 1) / 2) * spacing;
      const localY = (row - (rows - 1) / 2) * spacing;
      
      // Rotate around center
      const rotatedX = localX * Math.cos(angle) - localY * Math.sin(angle);
      const rotatedY = localX * Math.sin(angle) + localY * Math.cos(angle);
      
      positions.push({
        x: center.x + rotatedX,
        y: center.y + rotatedY,
      });
    }
    return positions;
  }

  /**
   * Calculate positions for wedge formation.
   * Units are arranged in a triangular wedge pointing in facing direction.
   * Row width increases: 1, 2, 3, 4, ...
   * @param center - Center point of the formation
   * @param count - Number of units
   * @param spacing - Distance between units
   * @param angle - Facing direction in radians
   */
  private calculateWedgeFormation(center: Vector2, count: number, spacing: number, angle: number): Vector2[] {
    const positions: Vector2[] = [];
    let row = 0;
    let remaining = count;
    
    while (remaining > 0) {
      const unitsInRow = Math.min(row + 1, remaining);
      for (let i = 0; i < unitsInRow; i++) {
        const localX = (i - (unitsInRow - 1) / 2) * spacing;
        const localY = -row * spacing; // Negative Y for forward-pointing wedge
        
        // Rotate around center
        const rotatedX = localX * Math.cos(angle) - localY * Math.sin(angle);
        const rotatedY = localX * Math.sin(angle) + localY * Math.cos(angle);
        
        positions.push({
          x: center.x + rotatedX,
          y: center.y + rotatedY,
        });
      }
      remaining -= unitsInRow;
      row++;
    }
    return positions;
  }

  private calculateGridFormation(center: Vector2, count: number, spacing: number): Vector2[] {
    const positions: Vector2[] = [];
    const cols = Math.ceil(Math.sqrt(count));
    
    for (let i = 0; i < count; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      positions.push({
        x: center.x + (col - (cols - 1) / 2) * spacing,
        y: center.y + (row - Math.floor((count - 1) / cols) / 2) * spacing,
      });
    }
    return positions;
  }

  // Animation update - move units toward their targets
  updateUnits(deltaTime: number) {
    const moveSpeed = 100; // pixels per second
    let needsUpdate = false;

    this.state.units.forEach(unit => {
      if (unit.targetPosition) {
        const dx = unit.targetPosition.x - unit.position.x;
        const dy = unit.targetPosition.y - unit.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 2) {
          unit.position = unit.targetPosition;
          unit.targetPosition = null;
          unit.path = [];
          needsUpdate = true;
        } else {
          const moveDistance = Math.min(moveSpeed * deltaTime, distance);
          unit.position.x += (dx / distance) * moveDistance;
          unit.position.y += (dy / distance) * moveDistance;
          needsUpdate = true;
        }
      }
    });

    if (needsUpdate) {
      this.notify();
    }
  }

  getGroupPath(): Vector2[] | null {
    if (!this.state.formationSettings.showGroupPath) return null;
    
    const selectedUnits = this.state.units.filter(u => u.selected);
    if (selectedUnits.length === 0) return null;

    // Calculate center of selected units
    const center = selectedUnits.reduce(
      (acc, u) => ({ x: acc.x + u.position.x, y: acc.y + u.position.y }),
      { x: 0, y: 0 }
    );
    center.x /= selectedUnits.length;
    center.y /= selectedUnits.length;

    // Return simple path to first target if any unit has a target
    const firstTarget = selectedUnits.find(u => u.targetPosition)?.targetPosition;
    if (firstTarget) {
      return [center, firstTarget];
    }
    return null;
  }
}

export const gameStore = new GameStore();
