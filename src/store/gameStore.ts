/**
 * Game Store - Manages game state for the RTS
 * Handles rally points, units, buildings, and resources
 */

export interface Resource {
  id: string;
  type: 'gold' | 'wood' | 'stone';
  position: { x: number; y: number };
  amount: number;
}

export interface RallyPoint {
  position: { x: number; y: number };
  targetResourceId?: string; // If set, rally point is on a resource
  targetResource?: Resource; // Reference to the resource
}

export interface Unit {
  id: string;
  type: 'worker' | 'soldier';
  position: { x: number; y: number };
  isGathering: boolean;
  targetResourceId?: string;
  playerId: string;
}

export interface Building {
  id: string;
  type: 'townhall' | 'barracks';
  position: { x: number; y: number };
  rallyPoint?: RallyPoint;
  playerId: string;
  spawnQueue: string[]; // Unit types to spawn
}

interface GameState {
  buildings: Map<string, Building>;
  units: Map<string, Unit>;
  resources: Map<string, Resource>;
  selectedBuildingId: string | null;
}

class GameStore {
  private state: GameState = {
    buildings: new Map(),
    units: new Map(),
    resources: new Map(),
    selectedBuildingId: null,
  };

  private listeners: Set<() => void> = new Set();

  /**
   * Subscribe to state changes
   */
  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify all listeners of state change
   */
  private notify() {
    this.listeners.forEach(listener => listener());
  }

  /**
   * Get current state
   */
  getState(): GameState {
    return this.state;
  }

  /**
   * Add a building to the game
   */
  addBuilding(building: Building) {
    this.state.buildings.set(building.id, building);
    this.notify();
  }

  /**
   * Add a unit to the game
   */
  addUnit(unit: Unit) {
    this.state.units.set(unit.id, unit);
    this.notify();
  }

  /**
   * Add a resource to the game
   */
  addResource(resource: Resource) {
    this.state.resources.set(resource.id, resource);
    this.notify();
  }

  /**
   * Set rally point for a building
   * If the position is on a resource, automatically set targetResourceId
   */
  setRallyPoint(buildingId: string, position: { x: number; y: number }) {
    const building = this.state.buildings.get(buildingId);
    if (!building) return;

    // Check if rally point is on a resource
    const resource = this.findResourceAtPosition(position);

    building.rallyPoint = {
      position,
      targetResourceId: resource?.id,
      targetResource: resource,
    };

    this.state.buildings.set(buildingId, building);
    this.notify();

    console.log(`Rally point set for building ${buildingId}`, {
      position,
      onResource: !!resource,
      resourceType: resource?.type,
    });
  }

  /**
   * Find a resource at the given position (with some tolerance)
   */
  private findResourceAtPosition(position: { x: number; y: number }): Resource | undefined {
    const RESOURCE_RADIUS = 50; // Detection radius for resources

    for (const resource of this.state.resources.values()) {
      const distance = Math.sqrt(
        Math.pow(resource.position.x - position.x, 2) +
        Math.pow(resource.position.y - position.y, 2)
      );

      if (distance <= RESOURCE_RADIUS) {
        return resource;
      }
    }

    return undefined;
  }

  /**
   * Spawn a unit from a building
   * If building has a rally point on a resource, assign the unit to gather
   */
  spawnUnit(buildingId: string, unitType: string = 'worker'): Unit | null {
    const building = this.state.buildings.get(buildingId);
    if (!building) return null;

    const unit: Unit = {
      id: `unit-${Date.now()}-${Math.random()}`,
      type: unitType as 'worker' | 'soldier',
      position: { ...building.position },
      isGathering: false,
      playerId: building.playerId,
    };

    // If rally point is set on a resource, assign worker to gather
    if (building.rallyPoint?.targetResourceId && unitType === 'worker') {
      unit.targetResourceId = building.rallyPoint.targetResourceId;
      unit.isGathering = true;
      unit.position = { ...building.rallyPoint.position };

      console.log(`Worker spawned and assigned to gather from resource ${unit.targetResourceId}`);
    } else if (building.rallyPoint) {
      // Otherwise just move to rally point
      unit.position = { ...building.rallyPoint.position };
    }

    this.addUnit(unit);
    return unit;
  }

  /**
   * Select a building
   */
  selectBuilding(buildingId: string | null) {
    this.state.selectedBuildingId = buildingId;
    this.notify();
  }

  /**
   * Get selected building
   */
  getSelectedBuilding(): Building | null {
    if (!this.state.selectedBuildingId) return null;
    return this.state.buildings.get(this.state.selectedBuildingId) || null;
  }

  /**
   * Clear rally point for a building
   */
  clearRallyPoint(buildingId: string) {
    const building = this.state.buildings.get(buildingId);
    if (!building) return;

    building.rallyPoint = undefined;
    this.state.buildings.set(buildingId, building);
    this.notify();
  }
}

// Singleton instance
export const gameStore = new GameStore();
