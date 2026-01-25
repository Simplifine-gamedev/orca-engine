/**
 * Wall Store - Manages wall and gate states for the RTS game
 */

export interface Position {
  x: number;
  y: number;
}

export interface Gate {
  id: string;
  position: Position;
  isOpen: boolean;
  ownerId: string; // Player/faction that owns this gate
  closeTimer?: NodeJS.Timeout;
}

export interface Wall {
  id: string;
  position: Position;
  ownerId: string;
}

export interface Unit {
  id: string;
  position: Position;
  ownerId: string; // Player/faction this unit belongs to
  path?: Position[];
}

interface WallStoreState {
  gates: Map<string, Gate>;
  walls: Map<string, Wall>;
  units: Map<string, Unit>;
}

class WallStore {
  private state: WallStoreState = {
    gates: new Map(),
    walls: new Map(),
    units: new Map(),
  };

  private listeners: Set<() => void> = new Set();

  // Gate management
  addGate(gate: Gate): void {
    this.state.gates.set(gate.id, gate);
    this.notifyListeners();
  }

  removeGate(gateId: string): void {
    const gate = this.state.gates.get(gateId);
    if (gate?.closeTimer) {
      clearTimeout(gate.closeTimer);
    }
    this.state.gates.delete(gateId);
    this.notifyListeners();
  }

  getGate(gateId: string): Gate | undefined {
    return this.state.gates.get(gateId);
  }

  getAllGates(): Gate[] {
    return Array.from(this.state.gates.values());
  }

  // Open gate for friendly units
  openGate(gateId: string): void {
    const gate = this.state.gates.get(gateId);
    if (!gate) return;

    // Clear existing close timer if any
    if (gate.closeTimer) {
      clearTimeout(gate.closeTimer);
    }

    gate.isOpen = true;
    this.state.gates.set(gateId, gate);
    this.notifyListeners();
  }

  // Schedule gate to close after delay
  scheduleGateClose(gateId: string, delayMs: number = 2000): void {
    const gate = this.state.gates.get(gateId);
    if (!gate) return;

    // Clear existing timer
    if (gate.closeTimer) {
      clearTimeout(gate.closeTimer);
    }

    // Set new timer
    gate.closeTimer = setTimeout(() => {
      this.closeGate(gateId);
    }, delayMs);

    this.state.gates.set(gateId, gate);
  }

  // Close gate
  closeGate(gateId: string): void {
    const gate = this.state.gates.get(gateId);
    if (!gate) return;

    if (gate.closeTimer) {
      clearTimeout(gate.closeTimer);
      gate.closeTimer = undefined;
    }

    gate.isOpen = false;
    this.state.gates.set(gateId, gate);
    this.notifyListeners();
  }

  // Check if unit is friendly to gate owner
  isFriendlyUnit(unitOwnerId: string, gateOwnerId: string): boolean {
    // Simple ownership check - can be extended for alliances
    return unitOwnerId === gateOwnerId;
  }

  // Get gates near a position
  getGatesNearPosition(position: Position, radius: number): Gate[] {
    const nearbyGates: Gate[] = [];
    
    for (const gate of this.state.gates.values()) {
      const distance = Math.sqrt(
        Math.pow(gate.position.x - position.x, 2) +
        Math.pow(gate.position.y - position.y, 2)
      );
      
      if (distance <= radius) {
        nearbyGates.push(gate);
      }
    }
    
    return nearbyGates;
  }

  // Unit management
  addUnit(unit: Unit): void {
    this.state.units.set(unit.id, unit);
    this.notifyListeners();
  }

  removeUnit(unitId: string): void {
    this.state.units.delete(unitId);
    this.notifyListeners();
  }

  getUnit(unitId: string): Unit | undefined {
    return this.state.units.get(unitId);
  }

  updateUnitPosition(unitId: string, position: Position): void {
    const unit = this.state.units.get(unitId);
    if (!unit) return;

    unit.position = position;
    this.state.units.set(unitId, unit);
    this.notifyListeners();
  }

  getAllUnits(): Unit[] {
    return Array.from(this.state.units.values());
  }

  // Wall management
  addWall(wall: Wall): void {
    this.state.walls.set(wall.id, wall);
    this.notifyListeners();
  }

  removeWall(wallId: string): void {
    this.state.walls.delete(wallId);
    this.notifyListeners();
  }

  getAllWalls(): Wall[] {
    return Array.from(this.state.walls.values());
  }

  // Subscribe to store changes
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notifyListeners(): void {
    this.listeners.forEach((listener) => listener());
  }

  // Reset store (useful for testing)
  reset(): void {
    // Clear all timers
    for (const gate of this.state.gates.values()) {
      if (gate.closeTimer) {
        clearTimeout(gate.closeTimer);
      }
    }

    this.state = {
      gates: new Map(),
      walls: new Map(),
      units: new Map(),
    };
    this.notifyListeners();
  }
}

// Export singleton instance
export const wallStore = new WallStore();
