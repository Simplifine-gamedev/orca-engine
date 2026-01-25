// Simple test file for gameStore logic
// This can be run with a test runner like Jest or Vitest

import { Unit, UnitType } from '../types/game';

// Mock React for testing
const mockReact = {
  useReducer: () => [0, () => {}],
  useEffect: () => {},
};

(global as any).React = mockReact;

// Import after mocking
import { gameStore } from './gameStore';

// Test helper to reset store
const resetStore = () => {
  gameStore.setState({
    units: [],
    selectedUnits: [],
    resources: {
      wood: 100,
      food: 100,
      gold: 50,
      stone: 50,
    },
  });
};

// Test 1: Get idle workers
console.log('Test 1: Get idle workers');
resetStore();

const testUnits: Unit[] = [
  {
    id: 'worker-1',
    position: { x: 100, y: 100 },
    type: UnitType.WORKER,
    isSelected: false,
    isIdle: true,
  },
  {
    id: 'worker-2',
    position: { x: 150, y: 100 },
    type: UnitType.WORKER,
    isSelected: false,
    isIdle: false,
  },
  {
    id: 'soldier-1',
    position: { x: 200, y: 100 },
    type: UnitType.SOLDIER,
    isSelected: false,
    isIdle: true, // Idle soldier should not be counted
  },
];

testUnits.forEach(unit => gameStore.addUnit(unit));

const idleWorkers = gameStore.getIdleWorkers();
const idleWorkerCount = gameStore.getIdleWorkerCount();

console.assert(idleWorkers.length === 1, `Expected 1 idle worker, got ${idleWorkers.length}`);
console.assert(idleWorkerCount === 1, `Expected count 1, got ${idleWorkerCount}`);
console.assert(idleWorkers[0].id === 'worker-1', `Expected worker-1, got ${idleWorkers[0].id}`);
console.log('✓ Test 1 passed');

// Test 2: Select idle workers
console.log('\nTest 2: Select idle workers');
gameStore.selectIdleWorkers();

const state = gameStore.getState();
console.assert(state.selectedUnits.length === 1, `Expected 1 selected unit, got ${state.selectedUnits.length}`);
console.assert(state.selectedUnits[0] === 'worker-1', `Expected worker-1 selected, got ${state.selectedUnits[0]}`);
console.log('✓ Test 2 passed');

// Test 3: Multiple idle workers
console.log('\nTest 3: Multiple idle workers');
resetStore();

const multipleWorkers: Unit[] = [
  {
    id: 'worker-1',
    position: { x: 100, y: 100 },
    type: UnitType.WORKER,
    isSelected: false,
    isIdle: true,
  },
  {
    id: 'worker-2',
    position: { x: 150, y: 100 },
    type: UnitType.WORKER,
    isSelected: false,
    isIdle: true,
  },
  {
    id: 'worker-3',
    position: { x: 200, y: 100 },
    type: UnitType.WORKER,
    isSelected: false,
    isIdle: true,
  },
];

multipleWorkers.forEach(unit => gameStore.addUnit(unit));

const multipleIdleWorkers = gameStore.getIdleWorkers();
console.assert(multipleIdleWorkers.length === 3, `Expected 3 idle workers, got ${multipleIdleWorkers.length}`);

gameStore.selectIdleWorkers();
const multipleState = gameStore.getState();
console.assert(multipleState.selectedUnits.length === 3, `Expected 3 selected units, got ${multipleState.selectedUnits.length}`);
console.log('✓ Test 3 passed');

// Test 4: No idle workers
console.log('\nTest 4: No idle workers');
resetStore();

const busyWorkers: Unit[] = [
  {
    id: 'worker-1',
    position: { x: 100, y: 100 },
    type: UnitType.WORKER,
    isSelected: false,
    isIdle: false,
  },
];

busyWorkers.forEach(unit => gameStore.addUnit(unit));

const noIdleWorkers = gameStore.getIdleWorkers();
console.assert(noIdleWorkers.length === 0, `Expected 0 idle workers, got ${noIdleWorkers.length}`);

gameStore.selectIdleWorkers();
const noIdleState = gameStore.getState();
console.assert(noIdleState.selectedUnits.length === 0, `Expected 0 selected units, got ${noIdleState.selectedUnits.length}`);
console.log('✓ Test 4 passed');

console.log('\n✅ All tests passed!');

export {};
