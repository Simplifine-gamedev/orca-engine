/**
 * RTS Pathfinding System
 * Implements A* pathfinding with improvements for obstacle avoidance,
 * group pathfinding, dynamic obstacles, and path smoothing
 */

export interface Vector2 {
	x: number;
	y: number;
}

export interface PathNode {
	x: number;
	y: number;
	g: number; // Cost from start
	h: number; // Heuristic cost to goal
	f: number; // Total cost (g + h)
	parent: PathNode | null;
}

export interface Obstacle {
	x: number;
	y: number;
	radius: number;
	isStatic: boolean;
	isDynamic: boolean;
	constructionState?: number; // 0-1, for buildings under construction
}

export interface PathfindingOptions {
	smoothPath?: boolean;
	avoidDynamicObstacles?: boolean;
	groupPathfinding?: boolean;
	maxIterations?: number;
	diagonalMovement?: boolean;
	heuristicWeight?: number;
}

export interface GridCell {
	x: number;
	y: number;
	walkable: boolean;
	cost: number;
}

export class PathfindingGrid {
	private width: number;
	private height: number;
	private cellSize: number;
	private grid: GridCell[][];
	private obstacles: Map<string, Obstacle>;
	private dynamicObstacles: Set<string>;
	private cacheHash: string;
	private pathCache: Map<string, Vector2[]>;

	constructor(width: number, height: number, cellSize: number = 1) {
		this.width = Math.floor(width / cellSize);
		this.height = Math.floor(height / cellSize);
		this.cellSize = cellSize;
		this.obstacles = new Map();
		this.dynamicObstacles = new Set();
		this.pathCache = new Map();
		this.cacheHash = '';
		this.initializeGrid();
	}

	private initializeGrid(): void {
		this.grid = [];
		for (let y = 0; y < this.height; y++) {
			this.grid[y] = [];
			for (let x = 0; x < this.width; x++) {
				this.grid[y][x] = {
					x,
					y,
					walkable: true,
					cost: 1
				};
			}
		}
	}

	/**
	 * Add obstacle with construction state support (fix from commit 5434cdc)
	 */
	addObstacle(obstacle: Obstacle): void {
		const key = this.getObstacleKey(obstacle);
		this.obstacles.set(key, obstacle);
		
		if (obstacle.isDynamic) {
			this.dynamicObstacles.add(key);
		}

		this.updateGridAroundObstacle(obstacle);
		this.invalidateCache();
	}

	removeObstacle(obstacle: Obstacle): void {
		const key = this.getObstacleKey(obstacle);
		this.obstacles.delete(key);
		this.dynamicObstacles.delete(key);
		this.updateGridAroundObstacle(obstacle, true);
		this.invalidateCache();
	}

	/**
	 * Update obstacle (for dynamic obstacles and construction state changes)
	 */
	updateObstacle(obstacle: Obstacle): void {
		const key = this.getObstacleKey(obstacle);
		if (this.obstacles.has(key)) {
			this.obstacles.set(key, obstacle);
			this.updateGridAroundObstacle(obstacle);
			this.invalidateCache();
		}
	}

	private getObstacleKey(obstacle: Obstacle): string {
		// Include construction state in hash (fix from commit 5434cdc)
		const constructionState = obstacle.constructionState ?? 1;
		return `${obstacle.x},${obstacle.y},${obstacle.radius},${constructionState}`;
	}

	private updateGridAroundObstacle(obstacle: Obstacle, remove: boolean = false): void {
		const radiusCells = Math.ceil(obstacle.radius / this.cellSize);
		const centerX = Math.floor(obstacle.x / this.cellSize);
		const centerY = Math.floor(obstacle.y / this.cellSize);

		for (let dy = -radiusCells; dy <= radiusCells; dy++) {
			for (let dx = -radiusCells; dx <= radiusCells; dx++) {
				const x = centerX + dx;
				const y = centerY + dy;

				if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
					const worldX = x * this.cellSize;
					const worldY = y * this.cellSize;
					const dist = Math.sqrt(
						Math.pow(worldX - obstacle.x, 2) + 
						Math.pow(worldY - obstacle.y, 2)
					);

					if (dist <= obstacle.radius) {
						if (remove) {
							this.grid[y][x].walkable = true;
							this.grid[y][x].cost = 1;
						} else {
							this.grid[y][x].walkable = false;
							// Partially constructed buildings have higher cost but may be walkable
							const constructionState = obstacle.constructionState ?? 1;
							if (constructionState < 1) {
								this.grid[y][x].walkable = true;
								this.grid[y][x].cost = 5; // Higher cost for construction areas
							}
						}
					} else if (dist <= obstacle.radius + 2) {
						// Add buffer zone around obstacles with higher cost
						this.grid[y][x].cost = 2;
					}
				}
			}
		}
	}

	private getCacheKey(): string {
		// Generate cache hash including dynamic obstacles and construction states
		const staticHash = Array.from(this.obstacles.entries())
			.filter(([key]) => !this.dynamicObstacles.has(key))
			.map(([key, obs]) => `${key},${obs.constructionState ?? 1}`)
			.sort()
			.join('|');

		const dynamicHash = Array.from(this.dynamicObstacles)
			.map(key => {
				const obs = this.obstacles.get(key);
				return obs ? `${key},${obs.constructionState ?? 1}` : '';
			})
			.sort()
			.join('|');

		return `${staticHash}::${dynamicHash}`;
	}

	private invalidateCache(): void {
		const newHash = this.getCacheKey();
		if (newHash !== this.cacheHash) {
			this.pathCache.clear();
			this.cacheHash = newHash;
		}
	}

	/**
	 * Main pathfinding method with improved A* algorithm
	 */
	findPath(
		start: Vector2,
		goal: Vector2,
		options: PathfindingOptions = {}
	): Vector2[] | null {
		const {
			smoothPath = true,
			avoidDynamicObstacles = true,
			maxIterations = 10000,
			diagonalMovement = true,
			heuristicWeight = 1.0
		} = options;

		// Check cache
		const cacheKey = `${start.x},${start.y}-${goal.x},${goal.y}-${this.cacheHash}`;
		if (this.pathCache.has(cacheKey)) {
			return this.pathCache.get(cacheKey)!;
		}

		const startNode = this.worldToGrid(start);
		const goalNode = this.worldToGrid(goal);

		// Validate start and goal
		if (!this.isWalkable(startNode.x, startNode.y) || 
		    !this.isWalkable(goalNode.x, goalNode.y)) {
			return null;
		}

		const openSet = new Set<string>();
		const closedSet = new Set<string>();
		const nodeMap = new Map<string, PathNode>();

		const startPathNode: PathNode = {
			x: startNode.x,
			y: startNode.y,
			g: 0,
			h: this.heuristic(startNode, goalNode, heuristicWeight),
			f: 0,
			parent: null
		};
		startPathNode.f = startPathNode.g + startPathNode.h;

		const startKey = this.nodeKey(startNode);
		openSet.add(startKey);
		nodeMap.set(startKey, startPathNode);

		let iterations = 0;

		while (openSet.size > 0 && iterations < maxIterations) {
			iterations++;

			// Find node with lowest f score
			const currentKey = this.getLowestFScore(openSet, nodeMap);
			const current = nodeMap.get(currentKey)!;

			// Check if we reached the goal
			if (current.x === goalNode.x && current.y === goalNode.y) {
				const path = this.reconstructPath(current);
				const worldPath = path.map(p => this.gridToWorld(p));
				
				if (smoothPath) {
					const smoothedPath = this.smoothPath(worldPath);
					this.pathCache.set(cacheKey, smoothedPath);
					return smoothedPath;
				}
				
				this.pathCache.set(cacheKey, worldPath);
				return worldPath;
			}

			openSet.delete(currentKey);
			closedSet.add(currentKey);

			// Check neighbors
			const neighbors = this.getNeighbors(current, diagonalMovement);

			for (const neighbor of neighbors) {
				const neighborKey = this.nodeKey(neighbor);

				if (closedSet.has(neighborKey)) {
					continue;
				}

				if (!this.isWalkable(neighbor.x, neighbor.y)) {
					continue;
				}

				const moveCost = this.getMoveCost(current, neighbor, avoidDynamicObstacles);
				const tentativeG = current.g + moveCost;

				let neighborNode = nodeMap.get(neighborKey);

				if (!neighborNode) {
					neighborNode = {
						x: neighbor.x,
						y: neighbor.y,
						g: Infinity,
						h: this.heuristic(neighbor, goalNode, heuristicWeight),
						f: Infinity,
						parent: null
					};
					nodeMap.set(neighborKey, neighborNode);
				}

				if (tentativeG < neighborNode.g) {
					neighborNode.parent = current;
					neighborNode.g = tentativeG;
					neighborNode.f = neighborNode.g + neighborNode.h;

					if (!openSet.has(neighborKey)) {
						openSet.add(neighborKey);
					}
				}
			}
		}

		// No path found
		return null;
	}

	/**
	 * Path smoothing using line-of-sight optimization
	 */
	private smoothPath(path: Vector2[]): Vector2[] {
		if (path.length <= 2) {
			return path;
		}

		const smoothed: Vector2[] = [path[0]];
		let current = 0;

		while (current < path.length - 1) {
			let farthest = current + 1;

			// Find the farthest point with line of sight
			for (let i = current + 2; i < path.length; i++) {
				if (this.hasLineOfSight(path[current], path[i])) {
					farthest = i;
				} else {
					break;
				}
			}

			smoothed.push(path[farthest]);
			current = farthest;
		}

		return smoothed;
	}

	/**
	 * Check if there's a clear line of sight between two points
	 */
	private hasLineOfSight(from: Vector2, to: Vector2): boolean {
		const dx = to.x - from.x;
		const dy = to.y - from.y;
		const distance = Math.sqrt(dx * dx + dy * dy);
		const steps = Math.ceil(distance / (this.cellSize * 0.5));

		for (let i = 0; i <= steps; i++) {
			const t = i / steps;
			const x = from.x + dx * t;
			const y = from.y + dy * t;
			const grid = this.worldToGrid({ x, y });

			if (!this.isWalkable(grid.x, grid.y)) {
				return false;
			}
		}

		return true;
	}

	/**
	 * Get movement cost between two nodes, accounting for dynamic obstacles
	 */
	private getMoveCost(from: PathNode, to: PathNode, avoidDynamicObstacles: boolean): number {
		const dx = Math.abs(to.x - from.x);
		const dy = Math.abs(to.y - from.y);
		const baseCost = (dx === 1 && dy === 1) ? 1.414 : 1.0; // Diagonal vs straight

		let cost = baseCost * this.grid[to.y][to.x].cost;

		// Add extra cost for cells near dynamic obstacles
		if (avoidDynamicObstacles) {
			const worldPos = this.gridToWorld(to);
			for (const key of this.dynamicObstacles) {
				const obstacle = this.obstacles.get(key);
				if (obstacle) {
					const dist = Math.sqrt(
						Math.pow(worldPos.x - obstacle.x, 2) + 
						Math.pow(worldPos.y - obstacle.y, 2)
					);
					if (dist < obstacle.radius + 3) {
						cost *= 1.5; // Higher cost near dynamic obstacles
					}
				}
			}
		}

		return cost;
	}

	private heuristic(a: Vector2, b: Vector2, weight: number): number {
		// Euclidean distance
		const dx = Math.abs(a.x - b.x);
		const dy = Math.abs(a.y - b.y);
		return weight * Math.sqrt(dx * dx + dy * dy);
	}

	private getNeighbors(node: PathNode, diagonal: boolean): Vector2[] {
		const neighbors: Vector2[] = [];
		const directions = [
			{ x: 0, y: -1 },  // North
			{ x: 1, y: 0 },   // East
			{ x: 0, y: 1 },   // South
			{ x: -1, y: 0 },  // West
		];

		if (diagonal) {
			directions.push(
				{ x: 1, y: -1 },  // NE
				{ x: 1, y: 1 },   // SE
				{ x: -1, y: 1 },  // SW
				{ x: -1, y: -1 }  // NW
			);
		}

		for (const dir of directions) {
			const x = node.x + dir.x;
			const y = node.y + dir.y;

			if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
				neighbors.push({ x, y });
			}
		}

		return neighbors;
	}

	private nodeKey(node: { x: number; y: number }): string {
		return `${node.x},${node.y}`;
	}

	private getLowestFScore(openSet: Set<string>, nodeMap: Map<string, PathNode>): string {
		let lowestKey = '';
		let lowestF = Infinity;

		for (const key of openSet) {
			const node = nodeMap.get(key)!;
			if (node.f < lowestF) {
				lowestF = node.f;
				lowestKey = key;
			}
		}

		return lowestKey;
	}

	private reconstructPath(node: PathNode): Vector2[] {
		const path: Vector2[] = [];
		let current: PathNode | null = node;

		while (current) {
			path.unshift({ x: current.x, y: current.y });
			current = current.parent;
		}

		return path;
	}

	private worldToGrid(pos: Vector2): Vector2 {
		return {
			x: Math.floor(pos.x / this.cellSize),
			y: Math.floor(pos.y / this.cellSize)
		};
	}

	private gridToWorld(pos: Vector2): Vector2 {
		return {
			x: pos.x * this.cellSize + this.cellSize / 2,
			y: pos.y * this.cellSize + this.cellSize / 2
		};
	}

	private isWalkable(x: number, y: number): boolean {
		if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
			return false;
		}
		return this.grid[y][x].walkable;
	}

	/**
	 * Clear path cache (useful for dynamic updates)
	 */
	clearCache(): void {
		this.pathCache.clear();
	}

	getGridSize(): { width: number; height: number } {
		return { width: this.width, height: this.height };
	}
}
