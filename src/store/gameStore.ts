// Game state management for Orca RTS
// This store manages the game state including units, buildings, and resources

import { UnitConfig, FactionConfig, getUnitConfig } from '../config/factions';

export interface Unit {
  id: string;
  configId: string;
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  ownerId: string;
  isSelected: boolean;
  state: 'idle' | 'moving' | 'attacking' | 'gathering';
}

export interface Player {
  id: string;
  name: string;
  factionId: string;
  resources: {
    gold: number;
    wood: number;
    food: number;
  };
  units: Unit[];
  buildings: any[]; // Can be expanded later
}

export interface GameState {
  players: Player[];
  currentPlayerId: string;
  gameTime: number;
  isPaused: boolean;
  selectedUnits: string[];
}

// Initial game state
let gameState: GameState = {
  players: [],
  currentPlayerId: '',
  gameTime: 0,
  isPaused: false,
  selectedUnits: [],
};

// Game store actions
export const gameStore = {
  // Get current game state
  getState(): GameState {
    return gameState;
  },

  // Initialize a new player
  addPlayer(playerId: string, playerName: string, factionId: string): void {
    const player: Player = {
      id: playerId,
      name: playerName,
      factionId,
      resources: {
        gold: 500,
        wood: 300,
        food: 200,
      },
      units: [],
      buildings: [],
    };
    gameState.players.push(player);
  },

  // Create a new unit
  createUnit(playerId: string, unitConfigId: string, position: { x: number; y: number }): Unit | null {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) {
      console.error(`Player ${playerId} not found`);
      return null;
    }

    const unitConfig = getUnitConfig(unitConfigId);
    if (!unitConfig) {
      console.error(`Unit config ${unitConfigId} not found`);
      return null;
    }

    // Check if player has enough resources
    if (!this.canAffordUnit(playerId, unitConfig)) {
      console.error(`Player ${playerId} cannot afford unit ${unitConfigId}`);
      return null;
    }

    // Deduct resources
    this.deductResources(playerId, unitConfig.cost);

    // Create unit
    const unit: Unit = {
      id: `unit_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      configId: unitConfigId,
      position,
      health: unitConfig.stats.health,
      maxHealth: unitConfig.stats.health,
      ownerId: playerId,
      isSelected: false,
      state: 'idle',
    };

    player.units.push(unit);
    console.log(`Created ${unitConfig.name} for player ${playerId}`);
    return unit;
  },

  // Check if player can afford a unit
  canAffordUnit(playerId: string, unitConfig: UnitConfig): boolean {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) return false;

    const cost = unitConfig.cost;
    return (
      (cost.gold === undefined || player.resources.gold >= cost.gold) &&
      (cost.wood === undefined || player.resources.wood >= cost.wood) &&
      (cost.food === undefined || player.resources.food >= cost.food)
    );
  },

  // Deduct resources from player
  deductResources(playerId: string, cost: { gold?: number; wood?: number; food?: number }): void {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) return;

    if (cost.gold) player.resources.gold -= cost.gold;
    if (cost.wood) player.resources.wood -= cost.wood;
    if (cost.food) player.resources.food -= cost.food;
  },

  // Add resources to player (for testing or cheats)
  addResources(playerId: string, resources: { gold?: number; wood?: number; food?: number }): void {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) return;

    if (resources.gold) player.resources.gold += resources.gold;
    if (resources.wood) player.resources.wood += resources.wood;
    if (resources.food) player.resources.food += resources.food;
  },

  // Get player's units
  getPlayerUnits(playerId: string): Unit[] {
    const player = gameState.players.find(p => p.id === playerId);
    return player ? player.units : [];
  },

  // Get unit by id
  getUnit(unitId: string): Unit | undefined {
    for (const player of gameState.players) {
      const unit = player.units.find(u => u.id === unitId);
      if (unit) return unit;
    }
    return undefined;
  },

  // Select units
  selectUnits(unitIds: string[]): void {
    // Deselect all units first
    for (const player of gameState.players) {
      for (const unit of player.units) {
        unit.isSelected = false;
      }
    }

    // Select specified units
    gameState.selectedUnits = unitIds;
    for (const unitId of unitIds) {
      const unit = this.getUnit(unitId);
      if (unit) unit.isSelected = true;
    }
  },

  // Move unit to position
  moveUnit(unitId: string, targetPosition: { x: number; y: number }): void {
    const unit = this.getUnit(unitId);
    if (!unit) return;

    unit.state = 'moving';
    // In a real implementation, this would start a movement animation/interpolation
    unit.position = targetPosition;
    unit.state = 'idle';
  },

  // Update game time
  tick(deltaTime: number): void {
    if (!gameState.isPaused) {
      gameState.gameTime += deltaTime;
    }
  },

  // Pause/unpause game
  setPaused(paused: boolean): void {
    gameState.isPaused = paused;
  },

  // Reset game state
  reset(): void {
    gameState = {
      players: [],
      currentPlayerId: '',
      gameTime: 0,
      isPaused: false,
      selectedUnits: [],
    };
  },
};

export default gameStore;
