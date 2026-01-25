'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { GameState, Resources, ResourceIncome, Building, Unit, BUILDING_TYPES } from './types';

interface GameContextType {
  gameState: GameState;
  canAfford: (cost: Record<string, number>) => boolean;
  buildBuilding: (buildingType: string) => boolean;
  trainUnit: (unitType: string) => boolean;
}

const GameContext = createContext<GameContextType | undefined>(undefined);

export function GameProvider({ children }: { children: ReactNode }) {
  const [gameState, setGameState] = useState<GameState>({
    resources: {
      gold: 500,
      wood: 500,
      stone: 300,
      food: 200
    },
    income: {
      gold: 2,
      wood: 3,
      stone: 1,
      food: 5
    },
    buildings: [],
    units: [],
    selectedBuilding: null
  });

  // Resource income every second
  useEffect(() => {
    const interval = setInterval(() => {
      setGameState(prev => ({
        ...prev,
        resources: {
          gold: prev.resources.gold + prev.income.gold,
          wood: prev.resources.wood + prev.income.wood,
          stone: prev.resources.stone + prev.income.stone,
          food: prev.resources.food + prev.income.food
        }
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const canAfford = (cost: Record<string, number>): boolean => {
    return Object.entries(cost).every(([resource, amount]) => {
      const currentAmount = gameState.resources[resource as keyof Resources];
      return currentAmount >= amount;
    });
  };

  const buildBuilding = (buildingType: string): boolean => {
    const building = BUILDING_TYPES[buildingType];
    if (!building || !canAfford(building.cost)) {
      return false;
    }

    setGameState(prev => {
      const newResources = { ...prev.resources };
      Object.entries(building.cost).forEach(([resource, amount]) => {
        if (amount) {
          newResources[resource as keyof Resources] -= amount;
        }
      });

      const newIncome = { ...prev.income };
      if (building.produces) {
        Object.entries(building.produces).forEach(([resource, amount]) => {
          if (amount) {
            newIncome[resource as keyof ResourceIncome] += amount;
          }
        });
      }

      return {
        ...prev,
        resources: newResources,
        income: newIncome,
        buildings: [...prev.buildings, { ...building }]
      };
    });

    return true;
  };

  const trainUnit = (unitType: string): boolean => {
    // Unit training logic would go here
    return false;
  };

  return (
    <GameContext.Provider value={{ gameState, canAfford, buildBuilding, trainUnit }}>
      {children}
    </GameContext.Provider>
  );
}

export function useGame() {
  const context = useContext(GameContext);
  if (context === undefined) {
    throw new Error('useGame must be used within a GameProvider');
  }
  return context;
}
