/**
 * Example usage of the Orca RTS Pathfinding System
 * 
 * This file demonstrates various pathfinding scenarios
 */

import {
  createPathfinder,
  createAsyncPathfinder,
  createProgressivePathfinder,
  createFlowFieldPathfinder,
  Vector2D
} from './index';

/**
 * Example 1: Basic pathfinding with obstacles
 */
function basicPathfindingExample() {
  console.log('=== Basic Pathfinding Example ===');
  
  const pathfinder = createPathfinder(1.0);

  // Add some obstacles
  pathfinder.addObstacle('wall1', {
    x: 10,
    y: 10,
    radius: 3,
    constructionState: 'complete'
  });

  pathfinder.addObstacle('wall2', {
    x: 15,
    y: 15,
    radius: 2,
    constructionState: 'complete'
  });

  // Find path
  const start: Vector2D = { x: 0, y: 0 };
  const goal: Vector2D = { x: 20, y: 20 };

  const path = pathfinder.findPath(start, goal, {
    unitRadius: 0.5,
    allowDiagonal: true,
    smoothPath: true,
    avoidanceWeight: 0.5
  });

  if (path) {
    console.log(`Path found with ${path.length} waypoints`);
    console.log('Path:', path);
  } else {
    console.log('No path found');
  }

  // Stats
  const stats = pathfinder.getStats();
  console.log('Stats:', stats);
  console.log('');
}

/**
 * Example 2: Dynamic obstacles
 */
function dynamicObstaclesExample() {
  console.log('=== Dynamic Obstacles Example ===');
  
  const pathfinder = createPathfinder(1.0);

  // Add dynamic obstacle (moving unit)
  pathfinder.addObstacle('enemy_unit', {
    x: 10,
    y: 10,
    radius: 0.5,
    isDynamic: true
  });

  // Initial path
  let path = pathfinder.findPath({ x: 0, y: 0 }, { x: 20, y: 20 });
  console.log('Initial path length:', path?.length);

  // Move the obstacle
  pathfinder.updateObstacle('enemy_unit', { x: 15, y: 15 });

  // Path is automatically recalculated (cache invalidated)
  path = pathfinder.findPath({ x: 0, y: 0 }, { x: 20, y: 20 });
  console.log('Path after obstacle moved:', path?.length);
  console.log('');
}

/**
 * Example 3: Group pathfinding
 */
function groupPathfindingExample() {
  console.log('=== Group Pathfinding Example ===');
  
  const pathfinder = createPathfinder(1.0);

  // Add obstacles
  pathfinder.addObstacle('obstacle1', { x: 10, y: 10, radius: 2 });

  // Define units
  const units = [
    { id: 'unit1', start: { x: 0, y: 0 }, goal: { x: 20, y: 20 } },
    { id: 'unit2', start: { x: 1, y: 0 }, goal: { x: 21, y: 20 } },
    { id: 'unit3', start: { x: 2, y: 0 }, goal: { x: 22, y: 20 } },
    { id: 'unit4', start: { x: 0, y: 1 }, goal: { x: 20, y: 21 } },
    { id: 'unit5', start: { x: 1, y: 1 }, goal: { x: 21, y: 21 } }
  ];

  const paths = pathfinder.findGroupPaths(units, {
    unitRadius: 0.5,
    smoothPath: true
  });

  console.log(`Paths found for ${paths.size} units:`);
  for (const [unitId, path] of paths.entries()) {
    console.log(`  ${unitId}: ${path ? `${path.length} waypoints` : 'No path'}`);
  }
  console.log('');
}

/**
 * Example 4: Async pathfinding
 */
async function asyncPathfindingExample() {
  console.log('=== Async Pathfinding Example ===');
  
  const pathfinder = createPathfinder(1.0);
  const asyncPathfinder = createAsyncPathfinder(pathfinder);

  // Configure performance
  asyncPathfinder.setProcessingParams(10, 16);

  // Add obstacles
  asyncPathfinder.addObstacle('building', { x: 10, y: 10, radius: 3 });

  // Request multiple paths asynchronously
  const pathPromises = [
    asyncPathfinder.requestPath('req1', { x: 0, y: 0 }, { x: 20, y: 20 }, { unitRadius: 0.5 }, 1),
    asyncPathfinder.requestPath('req2', { x: 5, y: 5 }, { x: 25, y: 25 }, { unitRadius: 0.5 }, 2),
    asyncPathfinder.requestPath('req3', { x: 10, y: 0 }, { x: 30, y: 20 }, { unitRadius: 0.5 }, 0)
  ];

  const paths = await Promise.all(pathPromises);
  
  console.log('Async paths received:');
  paths.forEach((path, index) => {
    console.log(`  Request ${index + 1}: ${path ? `${path.length} waypoints` : 'No path'}`);
  });

  const stats = asyncPathfinder.getStats();
  console.log('Async stats:', stats);
  console.log('');
}

/**
 * Example 5: Progressive pathfinding for large groups
 */
async function progressivePathfindingExample() {
  console.log('=== Progressive Pathfinding Example ===');
  
  const pathfinder = createPathfinder(1.0);
  const asyncPathfinder = createAsyncPathfinder(pathfinder);
  const progressivePathfinder = createProgressivePathfinder(asyncPathfinder);

  // Generate large group of units
  const units = [];
  for (let i = 0; i < 20; i++) {
    units.push({
      id: `unit${i}`,
      start: { x: i % 5, y: Math.floor(i / 5) },
      goal: { x: 20 + (i % 5), y: 20 + Math.floor(i / 5) }
    });
  }

  // Find paths progressively with progress callback
  const results = await progressivePathfinder.findGroupPathsProgressive(
    units,
    { unitRadius: 0.5 },
    (progress, completed, total) => {
      console.log(`Progress: ${completed}/${total} (${(progress * 100).toFixed(1)}%)`);
    }
  );

  console.log(`Progressive pathfinding complete: ${results.size} paths found`);
  console.log('');
}

/**
 * Example 6: Flow field pathfinding
 */
function flowFieldExample() {
  console.log('=== Flow Field Pathfinding Example ===');
  
  const pathfinder = createPathfinder(1.0);
  const flowFieldPathfinder = createFlowFieldPathfinder(pathfinder);

  // Add obstacle
  pathfinder.addObstacle('center_obstacle', { x: 50, y: 50, radius: 5 });

  // Generate flow field
  const goal: Vector2D = { x: 100, y: 100 };
  const bounds = { minX: 0, maxX: 120, minY: 0, maxY: 120 };
  
  console.log('Generating flow field...');
  const flowField = flowFieldPathfinder.generateFlowField(goal, bounds, 2.0, 0.5);
  console.log(`Flow field generated with ${flowField.size} vectors`);

  // Get directions for some positions
  const testPositions: Vector2D[] = [
    { x: 10, y: 10 },
    { x: 50, y: 30 },
    { x: 80, y: 80 }
  ];

  console.log('Flow directions:');
  for (const pos of testPositions) {
    const direction = flowFieldPathfinder.getFlowDirection(flowField, pos, 2.0);
    if (direction) {
      console.log(`  (${pos.x}, ${pos.y}) -> direction (${direction.x.toFixed(2)}, ${direction.y.toFixed(2)})`);
    }
  }
  console.log('');
}

/**
 * Example 7: Construction state integration
 */
function constructionStateExample() {
  console.log('=== Construction State Example ===');
  
  const pathfinder = createPathfinder(1.0);

  // Add building under construction
  pathfinder.addObstacle('building1', {
    x: 10,
    y: 10,
    radius: 4,
    constructionState: 'foundation'
  });

  // Find initial path
  let path = pathfinder.findPath({ x: 0, y: 0 }, { x: 20, y: 20 });
  console.log('Path with foundation:', path?.length);

  // Update construction state
  pathfinder.updateObstacle('building1', { x: 10, y: 10 }, 'walls');
  
  // Cache should be invalidated due to construction state change
  path = pathfinder.findPath({ x: 0, y: 0 }, { x: 20, y: 20 });
  console.log('Path with walls:', path?.length);

  // Complete construction
  pathfinder.updateObstacle('building1', { x: 10, y: 10 }, 'complete');
  
  path = pathfinder.findPath({ x: 0, y: 0 }, { x: 20, y: 20 });
  console.log('Path with completed building:', path?.length);
  console.log('');
}

/**
 * Run all examples
 */
async function runAllExamples() {
  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║   Orca RTS Pathfinding System - Examples          ║');
  console.log('╚════════════════════════════════════════════════════╝');
  console.log('');

  basicPathfindingExample();
  dynamicObstaclesExample();
  groupPathfindingExample();
  await asyncPathfindingExample();
  await progressivePathfindingExample();
  flowFieldExample();
  constructionStateExample();

  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║   All examples completed successfully!            ║');
  console.log('╚════════════════════════════════════════════════════╝');
}

// Run examples if this file is executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}

export {
  basicPathfindingExample,
  dynamicObstaclesExample,
  groupPathfindingExample,
  asyncPathfindingExample,
  progressivePathfindingExample,
  flowFieldExample,
  constructionStateExample,
  runAllExamples
};
