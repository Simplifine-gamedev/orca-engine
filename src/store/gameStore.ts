// Game State Management
import { FactionConfig, UnitConfig, factions, scoutUnit } from "../config/factions";

export interface PlayerResources {
  gold: number;
  wood: number;
  food: number;
}

export interface UnitInstance {
  id: string;
  unitType: UnitConfig;
  position: { x: number; y: number };
  health: number;
  ownerId: string;
}

export interface Building {
  id: string;
  type: string;
  position: { x: number; y: number };
  health: number;
  ownerId: string;
  productionQueue: UnitConfig[];
}

export interface GameState {
  players: Map<string, PlayerState>;
  units: Map<string, UnitInstance>;
  buildings: Map<string, Building>;
  currentTick: number;
  gameSpeed: number;
}

export interface PlayerState {
  id: string;
  name: string;
  faction: FactionConfig;
  resources: PlayerResources;
  units: string[]; // Unit IDs
  buildings: string[]; // Building IDs
}

class GameStore {
  private state: GameState;
  private listeners: Set<(state: GameState) => void>;

  constructor() {
    this.state = {
      players: new Map(),
      units: new Map(),
      buildings: new Map(),
      currentTick: 0,
      gameSpeed: 1.0,
    };
    this.listeners = new Set();
  }

  // Initialize game with players
  initializeGame(playerCount: number = 1): void {
    this.state.players.clear();
    this.state.units.clear();
    this.state.buildings.clear();
    this.state.currentTick = 0;

    for (let i = 0; i < playerCount; i++) {
      const playerId = `player_${i}`;
      const playerState: PlayerState = {
        id: playerId,
        name: `Player ${i + 1}`,
        faction: factions[0],
        resources: {
          gold: 500,
          wood: 300,
          food: 200,
        },
        units: [],
        buildings: [],
      };
      this.state.players.set(playerId, playerState);
    }

    this.notifyListeners();
  }

  // Get current game state
  getState(): GameState {
    return this.state;
  }

  // Subscribe to state changes
  subscribe(listener: (state: GameState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // Notify all listeners of state change
  private notifyListeners(): void {
    this.listeners.forEach((listener) => listener(this.state));
  }

  // Create a unit (including scout units)
  createUnit(
    playerId: string,
    unitType: UnitConfig,
    position: { x: number; y: number }
  ): string | null {
    const player = this.state.players.get(playerId);
    if (!player) return null;

    // Check if player has enough resources
    if (!this.canAffordUnit(player, unitType)) {
      console.log(`Player ${playerId} cannot afford ${unitType.name}`);
      return null;
    }

    // Deduct resources
    this.deductResources(player, unitType.cost);

    // Create unit instance
    const unitId = `unit_${Date.now()}_${Math.random()}`;
    const unitInstance: UnitInstance = {
      id: unitId,
      unitType: unitType,
      position: position,
      health: unitType.stats.health,
      ownerId: playerId,
    };

    this.state.units.set(unitId, unitInstance);
    player.units.push(unitId);

    console.log(`Created ${unitType.name} for ${playerId} at`, position);
    this.notifyListeners();
    return unitId;
  }

  // Create a scout unit specifically (convenience method)
  createScout(playerId: string, position: { x: number; y: number }): string | null {
    return this.createUnit(playerId, scoutUnit, position);
  }

  // Check if player can afford a unit
  private canAffordUnit(player: PlayerState, unit: UnitConfig): boolean {
    const { gold = 0, wood = 0, food = 0 } = unit.cost;
    return (
      player.resources.gold >= gold &&
      player.resources.wood >= wood &&
      player.resources.food >= food
    );
  }

  // Deduct resources for unit creation
  private deductResources(
    player: PlayerState,
    cost: { gold?: number; wood?: number; food?: number }
  ): void {
    player.resources.gold -= cost.gold || 0;
    player.resources.wood -= cost.wood || 0;
    player.resources.food -= cost.food || 0;
  }

  // Move a unit
  moveUnit(unitId: string, newPosition: { x: number; y: number }): boolean {
    const unit = this.state.units.get(unitId);
    if (!unit) return false;

    unit.position = newPosition;
    this.notifyListeners();
    return true;
  }

  // Get player resources
  getPlayerResources(playerId: string): PlayerResources | null {
    const player = this.state.players.get(playerId);
    return player ? player.resources : null;
  }

  // Add resources to player
  addResources(playerId: string, resources: Partial<PlayerResources>): void {
    const player = this.state.players.get(playerId);
    if (!player) return;

    player.resources.gold += resources.gold || 0;
    player.resources.wood += resources.wood || 0;
    player.resources.food += resources.food || 0;

    this.notifyListeners();
  }

  // Get all units for a player
  getPlayerUnits(playerId: string): UnitInstance[] {
    const player = this.state.players.get(playerId);
    if (!player) return [];

    return player.units
      .map((unitId) => this.state.units.get(unitId))
      .filter((unit): unit is UnitInstance => unit !== undefined);
  }

  // Get visible area for a unit (based on vision range)
  getVisibleArea(unitId: string): { x: number; y: number; radius: number } | null {
    const unit = this.state.units.get(unitId);
    if (!unit) return null;

    return {
      x: unit.position.x,
      y: unit.position.y,
      radius: unit.unitType.stats.visionRange,
    };
  }

  // Update game tick
  updateTick(): void {
    this.state.currentTick++;
    this.notifyListeners();
  }
}

// Export singleton instance
export const gameStore = new GameStore();

// Export for testing
export { GameStore };
