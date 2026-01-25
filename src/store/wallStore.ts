// Wall and Gate state management store
import { Gate, Wall, Unit, Position } from '../types';

export interface WallStoreState {
  walls: Map<string, Wall>;
  gates: Map<string, Gate>;
  units: Map<string, Unit>;
}

class WallStore {
  private state: WallStoreState;
  private listeners: Set<(state: WallStoreState) => void>;
  private gateCheckInterval: number | null = null;

  constructor() {
    this.state = {
      walls: new Map(),
      gates: new Map(),
      units: new Map(),
    };
    this.listeners = new Set();
  }

  // Subscribe to state changes
  subscribe(listener: (state: WallStoreState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // Notify all listeners of state change
  private notify(): void {
    this.listeners.forEach(listener => listener(this.state));
  }

  // Get current state
  getState(): WallStoreState {
    return this.state;
  }

  // Add a wall or gate
  addWall(wall: Wall): void {
    if (wall.type === 'gate') {
      const gate: Gate = {
        ...wall,
        type: 'gate',
        isOpen: false,
        closeDelay: 2000, // 2 seconds default
        detectionRadius: 3, // 3 tiles default
      };
      this.state.gates.set(wall.id, gate);
    } else {
      this.state.walls.set(wall.id, wall);
    }
    this.notify();
  }

  // Remove a wall or gate
  removeWall(id: string): void {
    this.state.walls.delete(id);
    this.state.gates.delete(id);
    this.notify();
  }

  // Add or update a unit
  updateUnit(unit: Unit): void {
    this.state.units.set(unit.id, unit);
    this.notify();
  }

  // Remove a unit
  removeUnit(id: string): void {
    this.state.units.delete(id);
    this.notify();
  }

  // Open a gate
  openGate(gateId: string): void {
    const gate = this.state.gates.get(gateId);
    if (gate) {
      gate.isOpen = true;
      gate.lastOpenedTime = Date.now();
      this.state.gates.set(gateId, gate);
      this.notify();
    }
  }

  // Close a gate
  closeGate(gateId: string): void {
    const gate = this.state.gates.get(gateId);
    if (gate) {
      gate.isOpen = false;
      this.state.gates.set(gateId, gate);
      this.notify();
    }
  }

  // Calculate distance between two positions
  private getDistance(pos1: Position, pos2: Position): number {
    const dx = pos1.x - pos2.x;
    const dy = pos1.y - pos2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Check if any friendly units are near a gate
  private checkUnitsNearGate(gate: Gate): boolean {
    for (const unit of this.state.units.values()) {
      if (unit.team === 'friendly') {
        const distance = this.getDistance(unit.position, gate.position);
        if (distance <= gate.detectionRadius) {
          return true;
        }
      }
    }
    return false;
  }

  // Auto-manage gates based on nearby units
  private updateGates(): void {
    const now = Date.now();

    for (const gate of this.state.gates.values()) {
      const hasNearbyFriendlyUnits = this.checkUnitsNearGate(gate);

      if (hasNearbyFriendlyUnits && !gate.isOpen) {
        // Open gate for friendly units
        this.openGate(gate.id);
      } else if (gate.isOpen && !hasNearbyFriendlyUnits) {
        // Close gate after delay if no units nearby
        if (gate.lastOpenedTime && now - gate.lastOpenedTime > gate.closeDelay) {
          this.closeGate(gate.id);
        }
      } else if (gate.isOpen && hasNearbyFriendlyUnits) {
        // Keep gate open by updating the last opened time
        gate.lastOpenedTime = now;
        this.state.gates.set(gate.id, gate);
      }
    }
  }

  // Start automatic gate checking
  startGateChecking(intervalMs: number = 100): void {
    if (this.gateCheckInterval !== null) {
      return; // Already running
    }

    this.gateCheckInterval = window.setInterval(() => {
      this.updateGates();
    }, intervalMs);
  }

  // Stop automatic gate checking
  stopGateChecking(): void {
    if (this.gateCheckInterval !== null) {
      clearInterval(this.gateCheckInterval);
      this.gateCheckInterval = null;
    }
  }

  // Get all gates
  getGates(): Gate[] {
    return Array.from(this.state.gates.values());
  }

  // Get all walls (excluding gates)
  getWalls(): Wall[] {
    return Array.from(this.state.walls.values());
  }

  // Check if a position is blocked by a wall or closed gate
  isPositionBlocked(position: Position, team: 'friendly' | 'enemy'): boolean {
    // Check walls
    for (const wall of this.state.walls.values()) {
      if (wall.position.x === position.x && wall.position.y === position.y) {
        return true;
      }
    }

    // Check gates
    for (const gate of this.state.gates.values()) {
      if (gate.position.x === position.x && gate.position.y === position.y) {
        // Enemy units cannot pass through gates (even if open)
        if (team === 'enemy') {
          return true;
        }
        // Friendly units can pass if gate is open
        return !gate.isOpen;
      }
    }

    return false;
  }

  // Reset the store
  reset(): void {
    this.stopGateChecking();
    this.state = {
      walls: new Map(),
      gates: new Map(),
      units: new Map(),
    };
    this.notify();
  }
}

// Create and export a singleton instance
export const wallStore = new WallStore();
export default wallStore;
