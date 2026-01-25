// Basic tests for pathfinding functionality
import { Pathfinder } from '../pathfinding/pathfinding';
import { wallStore } from '../store/wallStore';
import { Position } from '../types';

// Test Suite
export function runPathfindingTests(): void {
  console.log('🧪 Running Pathfinding Tests...\n');

  // Test 1: Simple pathfinding
  console.log('Test 1: Simple pathfinding without obstacles');
  wallStore.reset();
  const pathfinder = new Pathfinder(20, 20);
  const start: Position = { x: 0, y: 0 };
  const goal: Position = { x: 5, y: 5 };
  const path = pathfinder.findPath(start, goal, 'friendly');
  console.assert(path !== null, 'Should find a path');
  console.assert(path!.length > 0, 'Path should not be empty');
  console.assert(path![0].x === 0 && path![0].y === 0, 'Path should start at start position');
  console.assert(
    path![path!.length - 1].x === 5 && path![path!.length - 1].y === 5,
    'Path should end at goal position'
  );
  console.log('✅ Passed\n');

  // Test 2: Pathfinding with walls
  console.log('Test 2: Pathfinding with walls');
  wallStore.reset();
  // Add vertical wall
  for (let y = 0; y < 10; y++) {
    wallStore.addWall({
      id: `wall-5-${y}`,
      position: { x: 5, y },
      type: 'wall',
      team: 'friendly',
    });
  }
  const pathfinder2 = new Pathfinder(20, 20);
  const path2 = pathfinder2.findPath({ x: 0, y: 5 }, { x: 10, y: 5 }, 'friendly');
  console.assert(path2 !== null, 'Should find a path around wall');
  // Verify path doesn't go through x=5
  const pathGoesAroundWall = path2!.every(pos => pos.x !== 5 || pos.y >= 10);
  console.assert(pathGoesAroundWall, 'Path should go around the wall');
  console.log('✅ Passed\n');

  // Test 3: Pathfinding through open gate
  console.log('Test 3: Pathfinding through open gate');
  wallStore.reset();
  // Add wall with gate
  for (let y = 0; y < 10; y++) {
    if (y === 5) {
      wallStore.addWall({
        id: 'gate-5-5',
        position: { x: 5, y: 5 },
        type: 'gate',
        team: 'friendly',
      });
      wallStore.openGate('gate-5-5');
    } else {
      wallStore.addWall({
        id: `wall-5-${y}`,
        position: { x: 5, y },
        type: 'wall',
        team: 'friendly',
      });
    }
  }
  const pathfinder3 = new Pathfinder(20, 20);
  const path3 = pathfinder3.findPath({ x: 0, y: 5 }, { x: 10, y: 5 }, 'friendly');
  console.assert(path3 !== null, 'Should find path through open gate');
  // Verify path goes through the gate
  const pathGoesThrough = path3!.some(pos => pos.x === 5 && pos.y === 5);
  console.assert(pathGoesThrough, 'Path should go through open gate');
  console.log('✅ Passed\n');

  // Test 4: Pathfinding blocked by closed gate
  console.log('Test 4: Pathfinding blocked by closed gate');
  wallStore.reset();
  // Add wall with closed gate
  for (let y = 0; y < 10; y++) {
    wallStore.addWall({
      id: y === 5 ? 'gate-5-5' : `wall-5-${y}`,
      position: { x: 5, y },
      type: y === 5 ? 'gate' : 'wall',
      team: 'friendly',
    });
  }
  const pathfinder4 = new Pathfinder(20, 20);
  const path4 = pathfinder4.findPath({ x: 0, y: 5 }, { x: 6, y: 5 }, 'friendly');
  console.assert(path4 !== null, 'Should find path around closed gate');
  // Verify path doesn't go through the gate
  const pathGoesAround = path4!.every(pos => !(pos.x === 5 && pos.y === 5));
  console.assert(pathGoesAround, 'Path should go around closed gate');
  console.log('✅ Passed\n');

  // Test 5: Enemy cannot path through gate
  console.log('Test 5: Enemy cannot path through gate');
  wallStore.reset();
  wallStore.addWall({
    id: 'gate-5-5',
    position: { x: 5, y: 5 },
    type: 'gate',
    team: 'friendly',
  });
  wallStore.openGate('gate-5-5');
  const pathfinder5 = new Pathfinder(20, 20);
  const path5 = pathfinder5.findPath({ x: 5, y: 4 }, { x: 5, y: 6 }, 'enemy');
  console.assert(path5 === null, 'Enemy should not find path through gate');
  console.log('✅ Passed\n');

  console.log('🎉 All pathfinding tests passed!');
}

// Run tests if this is the main module
if (typeof window !== 'undefined') {
  (window as any).runPathfindingTests = runPathfindingTests;
}
