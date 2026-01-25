/**
 * Example usage of the RTS Pathfinding System
 */

import { PathfindingGrid, Obstacle } from './pathfinding';
import { AsyncPathfindingManager } from './pathfindingAsync';

// Example 1: Basic pathfinding
function basicPathfindingExample() {
	// Create a 100x100 grid with cell size of 1
	const grid = new PathfindingGrid(100, 100, 1);

	// Add static obstacles (buildings)
	const building: Obstacle = {
		x: 50,
		y: 50,
		radius: 5,
		isStatic: true,
		isDynamic: false,
		constructionState: 1.0 // Fully constructed
	};
	grid.addObstacle(building);

	// Add building under construction (can be walked through but with penalty)
	const constructionSite: Obstacle = {
		x: 30,
		y: 30,
		radius: 4,
		isStatic: false,
		isDynamic: false,
		constructionState: 0.5 // 50% constructed
	};
	grid.addObstacle(constructionSite);

	// Find path with smoothing enabled
	const path = grid.findPath(
		{ x: 10, y: 10 },
		{ x: 90, y: 90 },
		{
			smoothPath: true,
			avoidDynamicObstacles: true,
			diagonalMovement: true
		}
	);

	if (path) {
		console.log('Path found with', path.length, 'waypoints');
		return path;
	} else {
		console.log('No path found');
		return null;
	}
}

// Example 2: Dynamic obstacles (moving units)
function dynamicObstaclesExample() {
	const grid = new PathfindingGrid(100, 100, 1);

	// Add a moving enemy unit as dynamic obstacle
	const enemyUnit: Obstacle = {
		x: 50,
		y: 50,
		radius: 1,
		isStatic: false,
		isDynamic: true
	};
	grid.addObstacle(enemyUnit);

	// Find path that avoids dynamic obstacles
	const path = grid.findPath(
		{ x: 45, y: 50 },
		{ x: 55, y: 50 },
		{ avoidDynamicObstacles: true }
	);

	// Update enemy position
	enemyUnit.x = 52;
	enemyUnit.y = 51;
	grid.updateObstacle(enemyUnit);

	// Recalculate path with new obstacle position
	const updatedPath = grid.findPath(
		{ x: 45, y: 50 },
		{ x: 55, y: 50 },
		{ avoidDynamicObstacles: true }
	);

	return updatedPath;
}

// Example 3: Async pathfinding for single unit
async function asyncPathfindingExample() {
	const grid = new PathfindingGrid(100, 100, 1);
	const manager = new AsyncPathfindingManager(grid, 5);

	// Request path asynchronously
	const path = await manager.requestPath(
		{ x: 10, y: 10 },
		{ x: 90, y: 90 },
		{ smoothPath: true },
		1 // Priority
	);

	if (path) {
		console.log('Async path found:', path.length, 'waypoints');
	}

	return path;
}

// Example 4: Group pathfinding with formation
async function groupPathfindingExample() {
	const grid = new PathfindingGrid(200, 200, 1);
	const manager = new AsyncPathfindingManager(grid, 10);

	// Create a group of units
	const units = [];
	for (let i = 0; i < 20; i++) {
		units.push({
			id: `unit_${i}`,
			position: { x: 10 + i * 2, y: 10 }
		});
	}

	// Request group path with wedge formation
	const groupPaths = await manager.requestGroupPath({
		groupId: 'army_1',
		units: units,
		goal: { x: 180, y: 180 },
		formation: 'wedge',
		options: {
			smoothPath: true,
			avoidDynamicObstacles: true
		}
	});

	console.log('Group paths calculated for', groupPaths.size, 'units');

	// Access individual unit paths
	for (const [unitId, path] of groupPaths) {
		if (path) {
			console.log(`Unit ${unitId}: path with ${path.length} waypoints`);
		}
	}

	return groupPaths;
}

// Example 5: Large army pathfinding with flow fields
async function largeArmyExample() {
	const grid = new PathfindingGrid(500, 500, 1);
	const manager = new AsyncPathfindingManager(grid, 20);

	// Add some obstacles
	for (let i = 0; i < 10; i++) {
		grid.addObstacle({
			x: 100 + i * 30,
			y: 100 + Math.random() * 200,
			radius: 10,
			isStatic: true,
			isDynamic: false
		});
	}

	// Create a large army (100 units)
	const units = [];
	for (let i = 0; i < 100; i++) {
		const row = Math.floor(i / 10);
		const col = i % 10;
		units.push({
			id: `unit_${i}`,
			position: { x: 20 + col * 3, y: 20 + row * 3 }
		});
	}

	// Request paths for large group (will use flow field optimization)
	const groupPaths = await manager.requestGroupPath({
		groupId: 'large_army',
		units: units,
		goal: { x: 450, y: 450 },
		formation: 'box',
		options: {
			smoothPath: true
		}
	});

	console.log('Large army paths calculated:', groupPaths.size, 'units');

	return groupPaths;
}

// Example 6: Handling construction state changes
function constructionStateExample() {
	const grid = new PathfindingGrid(100, 100, 1);

	// Building starts construction
	const building: Obstacle = {
		x: 50,
		y: 50,
		radius: 5,
		isStatic: false,
		isDynamic: false,
		constructionState: 0.0 // Just started
	};
	grid.addObstacle(building);

	// Units can walk through at high cost
	const path1 = grid.findPath({ x: 45, y: 50 }, { x: 55, y: 50 });

	// Building progresses
	building.constructionState = 0.7;
	grid.updateObstacle(building);

	// Building completes
	building.constructionState = 1.0;
	building.isStatic = true;
	grid.updateObstacle(building);

	// Now units must path around
	const path2 = grid.findPath({ x: 45, y: 50 }, { x: 55, y: 50 });

	return { beforeCompletion: path1, afterCompletion: path2 };
}

// Export examples
export {
	basicPathfindingExample,
	dynamicObstaclesExample,
	asyncPathfindingExample,
	groupPathfindingExample,
	largeArmyExample,
	constructionStateExample
};
