/**
 * Async Pathfinding for RTS Games
 * Handles group pathfinding, async processing, and flow field pathfinding for large groups
 */

import {
	Vector2,
	PathfindingGrid,
	PathfindingOptions,
	Obstacle
} from './pathfinding';

export interface PathRequest {
	id: string;
	start: Vector2;
	goal: Vector2;
	options?: PathfindingOptions;
	priority?: number;
}

export interface PathResult {
	id: string;
	path: Vector2[] | null;
	success: boolean;
	timestamp: number;
}

export interface GroupPathRequest {
	groupId: string;
	units: Array<{ id: string; position: Vector2 }>;
	goal: Vector2;
	formation?: 'scatter' | 'line' | 'box' | 'wedge';
	options?: PathfindingOptions;
}

export interface FlowField {
	width: number;
	height: number;
	cellSize: number;
	directions: Map<string, Vector2>;
	costs: Map<string, number>;
}

/**
 * Async pathfinding manager with priority queue and batching
 */
export class AsyncPathfindingManager {
	private grid: PathfindingGrid;
	private requestQueue: PathRequest[];
	private processing: boolean;
	private maxRequestsPerFrame: number;
	private callbacks: Map<string, (result: PathResult) => void>;
	private groupRequests: Map<string, GroupPathRequest>;

	constructor(grid: PathfindingGrid, maxRequestsPerFrame: number = 5) {
		this.grid = grid;
		this.requestQueue = [];
		this.processing = false;
		this.maxRequestsPerFrame = maxRequestsPerFrame;
		this.callbacks = new Map();
		this.groupRequests = new Map();
	}

	/**
	 * Request a path asynchronously
	 */
	async requestPath(
		start: Vector2,
		goal: Vector2,
		options?: PathfindingOptions,
		priority: number = 0
	): Promise<Vector2[] | null> {
		const id = this.generateRequestId();

		return new Promise((resolve) => {
			const request: PathRequest = {
				id,
				start,
				goal,
				options,
				priority
			};

			this.callbacks.set(id, (result: PathResult) => {
				resolve(result.path);
			});

			this.addToQueue(request);
			this.processQueue();
		});
	}

	/**
	 * Request paths for a group of units with formation support
	 */
	async requestGroupPath(groupRequest: GroupPathRequest): Promise<Map<string, Vector2[] | null>> {
		const { groupId, units, goal, formation = 'scatter', options } = groupRequest;

		this.groupRequests.set(groupId, groupRequest);

		// For small groups, use individual pathfinding
		if (units.length <= 3) {
			return this.individualGroupPaths(units, goal, options);
		}

		// For larger groups, use flow field pathfinding
		return this.flowFieldGroupPaths(units, goal, formation, options);
	}

	/**
	 * Individual pathfinding for small groups
	 */
	private async individualGroupPaths(
		units: Array<{ id: string; position: Vector2 }>,
		goal: Vector2,
		options?: PathfindingOptions
	): Promise<Map<string, Vector2[] | null>> {
		const results = new Map<string, Vector2[] | null>();

		// Stagger the goal positions slightly to prevent clustering
		const goalPositions = this.generateFormationPositions(
			goal,
			units.length,
			'scatter'
		);

		const pathPromises = units.map(async (unit, index) => {
			const unitGoal = goalPositions[index] || goal;
			const path = await this.requestPath(
				unit.position,
				unitGoal,
				options,
				1 // Higher priority for group paths
			);
			return { id: unit.id, path };
		});

		const paths = await Promise.all(pathPromises);

		for (const { id, path } of paths) {
			results.set(id, path);
		}

		return results;
	}

	/**
	 * Flow field pathfinding for large groups
	 * This prevents units from all following the same path
	 */
	private async flowFieldGroupPaths(
		units: Array<{ id: string; position: Vector2 }>,
		goal: Vector2,
		formation: string,
		options?: PathfindingOptions
	): Promise<Map<string, Vector2[] | null>> {
		const results = new Map<string, Vector2[] | null>();

		// Generate flow field from goal
		const flowField = this.generateFlowField(goal);

		// Generate formation goal positions
		const formationPositions = this.generateFormationPositions(
			goal,
			units.length,
			formation
		);

		// Assign each unit a path based on flow field
		for (let i = 0; i < units.length; i++) {
			const unit = units[i];
			const unitGoal = formationPositions[i] || goal;

			// Follow flow field to goal
			const path = this.followFlowField(
				unit.position,
				unitGoal,
				flowField,
				options
			);

			results.set(unit.id, path);
		}

		return results;
	}

	/**
	 * Generate a flow field for the given goal position
	 * Flow fields allow many units to navigate efficiently toward a goal
	 */
	private generateFlowField(goal: Vector2): FlowField {
		const gridSize = this.grid.getGridSize();
		const flowField: FlowField = {
			width: gridSize.width,
			height: gridSize.height,
			cellSize: 1,
			directions: new Map(),
			costs: new Map()
		};

		// Use Dijkstra's algorithm to generate cost field
		const startCell = this.worldToGrid(goal);
		const openSet = new Set<string>();
		const costs = new Map<string, number>();

		const startKey = this.cellKey(startCell);
		costs.set(startKey, 0);
		openSet.add(startKey);

		while (openSet.size > 0) {
			// Find cell with lowest cost
			let currentKey = '';
			let lowestCost = Infinity;

			for (const key of openSet) {
				const cost = costs.get(key) || Infinity;
				if (cost < lowestCost) {
					lowestCost = cost;
					currentKey = key;
				}
			}

			if (!currentKey) break;

			const [x, y] = currentKey.split(',').map(Number);
			openSet.delete(currentKey);

			// Check neighbors
			const neighbors = this.getNeighborCells({ x, y });

			for (const neighbor of neighbors) {
				const neighborKey = this.cellKey(neighbor);
				const moveCost = this.getFlowFieldMoveCost({ x, y }, neighbor);
				const newCost = lowestCost + moveCost;

				const existingCost = costs.get(neighborKey) || Infinity;

				if (newCost < existingCost) {
					costs.set(neighborKey, newCost);
					openSet.add(neighborKey);

					// Calculate direction vector
					const direction = this.normalize({
						x: x - neighbor.x,
						y: y - neighbor.y
					});
					flowField.directions.set(neighborKey, direction);
				}
			}
		}

		flowField.costs = costs;
		return flowField;
	}

	/**
	 * Follow a flow field from start to goal
	 */
	private followFlowField(
		start: Vector2,
		goal: Vector2,
		flowField: FlowField,
		options?: PathfindingOptions
	): Vector2[] | null {
		const path: Vector2[] = [start];
		let current = start;
		const maxSteps = 1000;
		let steps = 0;

		const goalThreshold = 2.0; // Distance threshold to consider goal reached

		while (steps < maxSteps) {
			steps++;

			// Check if we reached the goal
			const distToGoal = this.distance(current, goal);
			if (distToGoal < goalThreshold) {
				path.push(goal);
				break;
			}

			const currentCell = this.worldToGrid(current);
			const cellKey = this.cellKey(currentCell);
			const direction = flowField.directions.get(cellKey);

			if (!direction) {
				// No direction available, fall back to direct pathfinding
				const fallbackPath = this.grid.findPath(current, goal, options);
				if (fallbackPath) {
					return [...path, ...fallbackPath.slice(1)];
				}
				return null;
			}

			// Move in the direction of the flow field
			const stepSize = 1.5;
			const next: Vector2 = {
				x: current.x + direction.x * stepSize,
				y: current.y + direction.y * stepSize
			};

			path.push(next);
			current = next;
		}

		// Smooth the path
		if (options?.smoothPath !== false) {
			return this.simplifyPath(path);
		}

		return path;
	}

	/**
	 * Generate formation positions around a goal point
	 */
	private generateFormationPositions(
		goal: Vector2,
		count: number,
		formation: string
	): Vector2[] {
		const positions: Vector2[] = [];
		const spacing = 2.0;

		switch (formation) {
			case 'scatter':
				// Random positions around goal
				for (let i = 0; i < count; i++) {
					const angle = (Math.PI * 2 * i) / count + Math.random() * 0.5;
					const distance = spacing + Math.random() * spacing;
					positions.push({
						x: goal.x + Math.cos(angle) * distance,
						y: goal.y + Math.sin(angle) * distance
					});
				}
				break;

			case 'line':
				// Line formation
				const lineLength = count * spacing;
				const startX = goal.x - lineLength / 2;
				for (let i = 0; i < count; i++) {
					positions.push({
						x: startX + i * spacing,
						y: goal.y
					});
				}
				break;

			case 'box':
				// Box formation
				const side = Math.ceil(Math.sqrt(count));
				for (let i = 0; i < count; i++) {
					const row = Math.floor(i / side);
					const col = i % side;
					positions.push({
						x: goal.x + (col - side / 2) * spacing,
						y: goal.y + (row - side / 2) * spacing
					});
				}
				break;

			case 'wedge':
				// Wedge/triangle formation
				let row = 0;
				let unitsInRow = 1;
				let unitsPlaced = 0;

				while (unitsPlaced < count) {
					const rowStart = -(unitsInRow - 1) * spacing / 2;
					for (let i = 0; i < unitsInRow && unitsPlaced < count; i++) {
						positions.push({
							x: goal.x + rowStart + i * spacing,
							y: goal.y + row * spacing
						});
						unitsPlaced++;
					}
					row++;
					unitsInRow++;
				}
				break;

			default:
				// Default to scatter
				for (let i = 0; i < count; i++) {
					positions.push(goal);
				}
		}

		return positions;
	}

	/**
	 * Add request to priority queue
	 */
	private addToQueue(request: PathRequest): void {
		this.requestQueue.push(request);
		// Sort by priority (higher priority first)
		this.requestQueue.sort((a, b) => (b.priority || 0) - (a.priority || 0));
	}

	/**
	 * Process the request queue
	 */
	private async processQueue(): Promise<void> {
		if (this.processing) {
			return;
		}

		this.processing = true;

		while (this.requestQueue.length > 0) {
			const batch = this.requestQueue.splice(0, this.maxRequestsPerFrame);

			for (const request of batch) {
				const path = this.grid.findPath(
					request.start,
					request.goal,
					request.options
				);

				const result: PathResult = {
					id: request.id,
					path,
					success: path !== null,
					timestamp: Date.now()
				};

				const callback = this.callbacks.get(request.id);
				if (callback) {
					callback(result);
					this.callbacks.delete(request.id);
				}
			}

			// Yield to other tasks
			await this.sleep(0);
		}

		this.processing = false;
	}

	/**
	 * Simplify path by removing unnecessary waypoints
	 */
	private simplifyPath(path: Vector2[]): Vector2[] {
		if (path.length <= 2) {
			return path;
		}

		const simplified: Vector2[] = [path[0]];
		let current = 0;

		while (current < path.length - 1) {
			let farthest = current + 1;

			// Find the farthest point we can reach in a straight line
			for (let i = current + 2; i < path.length; i++) {
				const distance = this.distance(path[current], path[i]);
				if (distance < 5.0) {
					farthest = i;
				}
			}

			simplified.push(path[farthest]);
			current = farthest;
		}

		return simplified;
	}

	// Utility functions

	private generateRequestId(): string {
		return `path_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
	}

	private worldToGrid(pos: Vector2): Vector2 {
		return {
			x: Math.floor(pos.x),
			y: Math.floor(pos.y)
		};
	}

	private cellKey(cell: Vector2): string {
		return `${cell.x},${cell.y}`;
	}

	private getNeighborCells(cell: Vector2): Vector2[] {
		return [
			{ x: cell.x - 1, y: cell.y },
			{ x: cell.x + 1, y: cell.y },
			{ x: cell.x, y: cell.y - 1 },
			{ x: cell.x, y: cell.y + 1 },
			{ x: cell.x - 1, y: cell.y - 1 },
			{ x: cell.x + 1, y: cell.y - 1 },
			{ x: cell.x - 1, y: cell.y + 1 },
			{ x: cell.x + 1, y: cell.y + 1 }
		];
	}

	private getFlowFieldMoveCost(from: Vector2, to: Vector2): number {
		const dx = Math.abs(to.x - from.x);
		const dy = Math.abs(to.y - from.y);
		return (dx === 1 && dy === 1) ? 1.414 : 1.0;
	}

	private normalize(vec: Vector2): Vector2 {
		const length = Math.sqrt(vec.x * vec.x + vec.y * vec.y);
		if (length === 0) {
			return { x: 0, y: 0 };
		}
		return {
			x: vec.x / length,
			y: vec.y / length
		};
	}

	private distance(a: Vector2, b: Vector2): number {
		const dx = a.x - b.x;
		const dy = a.y - b.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

	private sleep(ms: number): Promise<void> {
		return new Promise(resolve => setTimeout(resolve, ms));
	}

	/**
	 * Cancel a pending request
	 */
	cancelRequest(id: string): void {
		const index = this.requestQueue.findIndex(req => req.id === id);
		if (index !== -1) {
			this.requestQueue.splice(index, 1);
		}
		this.callbacks.delete(id);
	}

	/**
	 * Cancel all pending requests
	 */
	cancelAllRequests(): void {
		this.requestQueue = [];
		this.callbacks.clear();
		this.groupRequests.clear();
	}

	/**
	 * Get queue status
	 */
	getQueueStatus(): { pending: number; processing: boolean } {
		return {
			pending: this.requestQueue.length,
			processing: this.processing
		};
	}
}
