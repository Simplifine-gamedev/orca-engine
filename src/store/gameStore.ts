// Game store for RTS state management
import * as React from 'react';
import { GameState, Unit, UnitType } from '../types/game';

type Listener = () => void;

class GameStore {
  private state: GameState = {
    units: [],
    selectedUnits: [],
    resources: {
      wood: 100,
      food: 100,
      gold: 50,
      stone: 50,
    },
  };

  private listeners: Set<Listener> = new Set();

  getState = (): GameState => {
    return this.state;
  };

  setState = (newState: Partial<GameState>) => {
    this.state = { ...this.state, ...newState };
    this.notifyListeners();
  };

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private notifyListeners = () => {
    this.listeners.forEach((listener) => listener());
  };

  // Selector: Get all idle workers
  getIdleWorkers = (): Unit[] => {
    return this.state.units.filter(
      (unit) => unit.type === UnitType.WORKER && unit.isIdle
    );
  };

  // Selector: Get idle worker count
  getIdleWorkerCount = (): number => {
    return this.getIdleWorkers().length;
  };

  // Action: Select all idle workers
  selectIdleWorkers = () => {
    const idleWorkers = this.getIdleWorkers();
    const idleWorkerIds = idleWorkers.map((worker) => worker.id);

    // Deselect all units first
    const updatedUnits = this.state.units.map((unit) => ({
      ...unit,
      isSelected: idleWorkerIds.includes(unit.id),
    }));

    this.setState({
      units: updatedUnits,
      selectedUnits: idleWorkerIds,
    });

    // If there are idle workers, center camera on first one (optional - would need camera integration)
    if (idleWorkers.length > 0) {
      console.log(`Selected ${idleWorkers.length} idle workers`);
    }
  };

  // Action: Select units by IDs
  selectUnits = (unitIds: string[]) => {
    const updatedUnits = this.state.units.map((unit) => ({
      ...unit,
      isSelected: unitIds.includes(unit.id),
    }));

    this.setState({
      units: updatedUnits,
      selectedUnits: unitIds,
    });
  };

  // Action: Add unit
  addUnit = (unit: Unit) => {
    this.setState({
      units: [...this.state.units, unit],
    });
  };

  // Action: Update unit
  updateUnit = (unitId: string, updates: Partial<Unit>) => {
    const updatedUnits = this.state.units.map((unit) =>
      unit.id === unitId ? { ...unit, ...updates } : unit
    );

    this.setState({
      units: updatedUnits,
    });
  };

  // Action: Remove unit
  removeUnit = (unitId: string) => {
    const updatedUnits = this.state.units.filter((unit) => unit.id !== unitId);
    const updatedSelectedUnits = this.state.selectedUnits.filter(
      (id) => id !== unitId
    );

    this.setState({
      units: updatedUnits,
      selectedUnits: updatedSelectedUnits,
    });
  };

  // Action: Update resources
  updateResources = (resources: Partial<GameState['resources']>) => {
    this.setState({
      resources: { ...this.state.resources, ...resources },
    });
  };
}

export const gameStore = new GameStore();

// Export hooks for React components
export const useGameStore = () => {
  const [, forceUpdate] = React.useReducer((x) => x + 1, 0);

  React.useEffect(() => {
    const unsubscribe = gameStore.subscribe(() => {
      forceUpdate();
    });

    return unsubscribe;
  }, []);

  return gameStore.getState();
};
