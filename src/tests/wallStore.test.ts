// Basic tests for wallStore functionality
import { wallStore } from '../store/wallStore';
import { Gate, Unit, Position } from '../types';

// Test helper to create a gate
const createTestGate = (id: string, position: Position): Gate => ({
  id,
  position,
  type: 'gate',
  isOpen: false,
  team: 'friendly',
  closeDelay: 2000,
  detectionRadius: 3,
});

// Test helper to create a unit
const createTestUnit = (id: string, position: Position, team: 'friendly' | 'enemy'): Unit => ({
  id,
  position,
  team,
  isMoving: false,
});

// Test Suite
export function runTests(): void {
  console.log('🧪 Running WallStore Tests...\n');

  // Test 1: Add and retrieve gates
  console.log('Test 1: Add and retrieve gates');
  wallStore.reset();
  const gate1 = createTestGate('gate-1', { x: 5, y: 5 });
  wallStore.addWall(gate1);
  const gates = wallStore.getGates();
  console.assert(gates.length === 1, 'Should have 1 gate');
  console.assert(gates[0].id === 'gate-1', 'Gate ID should match');
  console.log('✅ Passed\n');

  // Test 2: Position blocking for closed gate
  console.log('Test 2: Position blocking for closed gate');
  wallStore.reset();
  const gate2 = createTestGate('gate-2', { x: 10, y: 10 });
  wallStore.addWall(gate2);
  const isBlocked = wallStore.isPositionBlocked({ x: 10, y: 10 }, 'friendly');
  console.assert(isBlocked === true, 'Closed gate should block friendly units');
  console.log('✅ Passed\n');

  // Test 3: Position blocking for open gate (friendly)
  console.log('Test 3: Position blocking for open gate (friendly)');
  wallStore.reset();
  const gate3 = createTestGate('gate-3', { x: 15, y: 15 });
  wallStore.addWall(gate3);
  wallStore.openGate('gate-3');
  const isBlockedOpen = wallStore.isPositionBlocked({ x: 15, y: 15 }, 'friendly');
  console.assert(isBlockedOpen === false, 'Open gate should not block friendly units');
  console.log('✅ Passed\n');

  // Test 4: Enemy units always blocked by gates
  console.log('Test 4: Enemy units always blocked by gates');
  wallStore.reset();
  const gate4 = createTestGate('gate-4', { x: 20, y: 20 });
  wallStore.addWall(gate4);
  wallStore.openGate('gate-4');
  const isBlockedEnemy = wallStore.isPositionBlocked({ x: 20, y: 20 }, 'enemy');
  console.assert(isBlockedEnemy === true, 'Gates should always block enemy units');
  console.log('✅ Passed\n');

  // Test 5: Unit detection near gate
  console.log('Test 5: Unit detection near gate');
  wallStore.reset();
  const gate5 = createTestGate('gate-5', { x: 10, y: 10 });
  wallStore.addWall(gate5);
  
  // Add a friendly unit nearby
  const unit1 = createTestUnit('unit-1', { x: 11, y: 10 }, 'friendly');
  wallStore.updateUnit(unit1);
  
  // Start gate checking
  wallStore.startGateChecking(50);
  
  // Wait for gate to open
  setTimeout(() => {
    const gates = wallStore.getGates();
    const updatedGate = gates.find(g => g.id === 'gate-5');
    console.assert(updatedGate?.isOpen === true, 'Gate should open with nearby friendly unit');
    console.log('✅ Passed\n');
    
    // Cleanup
    wallStore.stopGateChecking();
    wallStore.reset();
    
    console.log('🎉 All tests passed!');
  }, 200);
}

// Run tests if this is the main module
if (typeof window !== 'undefined') {
  (window as any).runWallStoreTests = runTests;
}
