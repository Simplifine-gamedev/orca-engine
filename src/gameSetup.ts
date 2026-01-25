import { useGameStore } from './store/gameStore';
import type { Faction, Unit } from './types';

/**
 * Initialize the game with factions and set the player faction
 * This is an example setup showing how to properly initialize the game
 */
export const initializeGame = () => {
  const store = useGameStore.getState();
  
  // Create player faction
  const playerFaction: Faction = {
    id: 'player',
    name: 'Player Faction',
    color: '#3B82F6', // Blue
    population: 0,
    maxPopulation: 200,
    units: [],
  };
  
  // Create enemy factions
  const enemyFaction1: Faction = {
    id: 'enemy1',
    name: 'Enemy Red',
    color: '#EF4444', // Red
    population: 0,
    maxPopulation: 200,
    units: [],
  };
  
  const enemyFaction2: Faction = {
    id: 'enemy2',
    name: 'Enemy Green',
    color: '#10B981', // Green
    population: 0,
    maxPopulation: 200,
    units: [],
  };
  
  // Initialize factions in the store
  useGameStore.setState({
    factions: {
      player: playerFaction,
      enemy1: enemyFaction1,
      enemy2: enemyFaction2,
    },
    playerFactionId: 'player', // Set player faction
  });
  
  // Add some initial units for demonstration
  addStartingUnits();
};

/**
 * Add starting units to each faction for demonstration
 */
const addStartingUnits = () => {
  const store = useGameStore.getState();
  
  // Add 5 worker units to player faction
  for (let i = 0; i < 5; i++) {
    const unit: Unit = {
      id: `player-worker-${i}`,
      factionId: 'player',
      type: 'worker',
      health: 50,
      maxHealth: 50,
    };
    store.addUnit(unit);
  }
  
  // Add 3 soldier units to player faction
  for (let i = 0; i < 3; i++) {
    const unit: Unit = {
      id: `player-soldier-${i}`,
      factionId: 'player',
      type: 'soldier',
      health: 100,
      maxHealth: 100,
    };
    store.addUnit(unit);
  }
  
  // Add units to enemy factions (these should NOT appear in player population)
  for (let i = 0; i < 10; i++) {
    const enemyUnit1: Unit = {
      id: `enemy1-unit-${i}`,
      factionId: 'enemy1',
      type: 'soldier',
      health: 100,
      maxHealth: 100,
    };
    store.addUnit(enemyUnit1);
    
    const enemyUnit2: Unit = {
      id: `enemy2-unit-${i}`,
      factionId: 'enemy2',
      type: 'soldier',
      health: 100,
      maxHealth: 100,
    };
    store.addUnit(enemyUnit2);
  }
  
  // At this point:
  // - Player faction has 8 units
  // - Enemy factions have 10 units each
  // - World population is 28 total
  // - ResourceBar should show "8 / 200" for player faction, NOT "28"
};
