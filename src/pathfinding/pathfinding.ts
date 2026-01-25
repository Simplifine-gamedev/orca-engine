/**
 * Advanced Pathfinding System for Orca RTS
 * 
 * Features:
 * - A* pathfinding with improved heuristics
 * - Dynamic obstacle handling
 * - Path smoothing
 * - Construction state cache integration
 * - Group pathfinding support
 */

export interface Vector2D {
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
  closed: boolean;
  opened: boolean;
}

export interface Obstacle {
  x: number;
  y: number;
  radius: number;
  isDynamic?: boolean;
  constructionState?: string; // Added for construction state tracking (commit 5434cdc)
}

export interface PathfindingOptions {
  unitRadius?: number;
  allowDiagonal?: boolean;
  heuristicWeight?: number;
  smoothPath?: boolean;
  avoidanceWeight?: number;
  maxIterations?: number;
}

export interface PathCache {
  path: Vector2D[];
  timestamp: number;
  obstacleHash: string;
}

/**
 * Main Pathfinding Class
 */
export class Pathfinder {
  private gridSize: number;
  private obstacles: Map<string, Obstacle>;
  private pathCache: Map<string, PathCache>;
  private dynamicObstacles: Set<string>;
  
  constructor(gridSize: number = 1.0) {
    this.gridSize = gridSize;
    this.obstacles = new Map();
    this.pathCache = new Map();
    this.dynamicObstacles = new Set();
  }

  /**
   * Add obstacle with construction state support (commit 5434cdc)
   */
  addObstacle(id: string, obstacle: Obstacle): void {
    this.obstacles.set(id, obstacle);
    if (obstacle.isDynamic) {
      this.dynamicObstacles.add(id);
    }
    // Invalidate affected cache entries
    this.invalidateCache();
  }

  /**
   * Remove obstacle
   */
  removeObstacle(id: string): void {
    this.obstacles.delete(id);
    this.dynamicObstacles.delete(id);
    this.invalidateCache();
  }

  /**
   * Update obstacle (for dynamic obstacles)
   */
  updateObstacle(id: string, newPosition: Vector2D, constructionState?: string): void {
    const obstacle = this.obstacles.get(id);
    if (obstacle) {
      obstacle.x = newPosition.x;
      obstacle.y = newPosition.y;
      if (constructionState !== undefined) {
        obstacle.constructionState = constructionState;
      }
      if (obstacle.isDynamic) {
        this.invalidateCache();
      }
    }
  }

  /**
   * Generate hash for cache key including construction states
   */
  private generateObstacleHash(): string {
    const sortedObstacles = Array.from(this.obstacles.entries())
      .sort((a, b) => a[0].localeCompare(b[0]));
    
    const hashComponents = sortedObstacles.map(([id, obs]) => {
      return `${id}:${obs.x.toFixed(1)},${obs.y.toFixed(1)},${obs.radius.toFixed(1)},${obs.constructionState || 'none'}`;
    });
    
    return hashComponents.join('|');
  }

  /**
   * Invalidate path cache
   */
  private invalidateCache(): void {
    this.pathCache.clear();
  }

  /**
   * Check if position collides with any obstacle
   */
  private isColliding(x: number, y: number, unitRadius: number): boolean {
    for (const obstacle of this.obstacles.values()) {
      const dx = x - obstacle.x;
      const dy = y - obstacle.y;
      const distSq = dx * dx + dy * dy;
      const minDist = unitRadius + obstacle.radius;
      if (distSq < minDist * minDist) {
        return true;
      }
    }
    return false;
  }

  /**
   * Calculate obstacle avoidance cost
   */
  private getObstacleAvoidanceCost(x: number, y: number, unitRadius: number, avoidanceWeight: number): number {
    let cost = 0;
    const avoidanceRadius = unitRadius * 3.0; // Check obstacles within 3x unit radius
    
    for (const obstacle of this.obstacles.values()) {
      const dx = x - obstacle.x;
      const dy = y - obstacle.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const combinedRadius = obstacle.radius + unitRadius;
      
      if (dist < avoidanceRadius && dist > combinedRadius) {
        // Add cost inversely proportional to distance
        const avoidanceFactor = (avoidanceRadius - dist) / (avoidanceRadius - combinedRadius);
        cost += avoidanceWeight * avoidanceFactor;
      }
    }
    
    return cost;
  }

  /**
   * Improved A* heuristic with obstacle awareness
   */
  private heuristic(a: Vector2D, b: Vector2D, unitRadius: number, avoidanceWeight: number): number {
    const dx = Math.abs(a.x - b.x);
    const dy = Math.abs(a.y - b.y);
    
    // Octile distance (better for diagonal movement)
    const D = 1.0;
    const D2 = Math.SQRT2;
    const h = D * (dx + dy) + (D2 - 2 * D) * Math.min(dx, dy);
    
    // Add obstacle avoidance cost
    const avoidanceCost = this.getObstacleAvoidanceCost(a.x, a.y, unitRadius, avoidanceWeight);
    
    return h + avoidanceCost;
  }

  /**
   * Get neighbors for a node
   */
  private getNeighbors(node: PathNode, allowDiagonal: boolean): Vector2D[] {
    const neighbors: Vector2D[] = [];
    const directions = [
      { x: 1, y: 0 },   // Right
      { x: -1, y: 0 },  // Left
      { x: 0, y: 1 },   // Down
      { x: 0, y: -1 },  // Up
    ];

    if (allowDiagonal) {
      directions.push(
        { x: 1, y: 1 },   // Down-Right
        { x: 1, y: -1 },  // Up-Right
        { x: -1, y: 1 },  // Down-Left
        { x: -1, y: -1 }  // Up-Left
      );
    }

    for (const dir of directions) {
      neighbors.push({
        x: node.x + dir.x * this.gridSize,
        y: node.y + dir.y * this.gridSize
      });
    }

    return neighbors;
  }

  /**
   * Smooth path using Catmull-Rom spline simplification
   */
  private smoothPath(path: Vector2D[], unitRadius: number): Vector2D[] {
    if (path.length <= 2) {
      return path;
    }

    const smoothed: Vector2D[] = [path[0]];
    let i = 0;

    while (i < path.length - 1) {
      let j = path.length - 1;
      let found = false;

      // Try to find the farthest visible point
      while (j > i + 1) {
        if (this.hasLineOfSight(path[i], path[j], unitRadius)) {
          smoothed.push(path[j]);
          i = j;
          found = true;
          break;
        }
        j--;
      }

      if (!found) {
        i++;
        if (i < path.length) {
          smoothed.push(path[i]);
        }
      }
    }

    return smoothed;
  }

  /**
   * Check line of sight between two points
   */
  private hasLineOfSight(a: Vector2D, b: Vector2D, unitRadius: number): boolean {
    const steps = Math.ceil(Math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2) / (this.gridSize * 0.5));
    
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = a.x + (b.x - a.x) * t;
      const y = a.y + (b.y - a.y) * t;
      
      if (this.isColliding(x, y, unitRadius)) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Find path using A* algorithm with improvements
   */
  findPath(start: Vector2D, goal: Vector2D, options: PathfindingOptions = {}): Vector2D[] | null {
    const {
      unitRadius = 0.5,
      allowDiagonal = true,
      heuristicWeight = 1.0,
      smoothPath = true,
      avoidanceWeight = 0.5,
      maxIterations = 10000
    } = options;

    // Check cache
    const cacheKey = `${start.x},${start.y}-${goal.x},${goal.y}-${unitRadius}`;
    const obstacleHash = this.generateObstacleHash();
    const cached = this.pathCache.get(cacheKey);
    
    if (cached && cached.obstacleHash === obstacleHash) {
      // Cache hit
      return cached.path;
    }

    // Check if start or goal is blocked
    if (this.isColliding(start.x, start.y, unitRadius) || this.isColliding(goal.x, goal.y, unitRadius)) {
      return null;
    }

    // Initialize start node
    const startNode: PathNode = {
      x: start.x,
      y: start.y,
      g: 0,
      h: this.heuristic(start, goal, unitRadius, avoidanceWeight) * heuristicWeight,
      f: 0,
      parent: null,
      closed: false,
      opened: true
    };
    startNode.f = startNode.g + startNode.h;

    const openList: PathNode[] = [startNode];
    const closedSet = new Set<string>();
    const nodeMap = new Map<string, PathNode>();
    nodeMap.set(`${start.x},${start.y}`, startNode);

    let iterations = 0;

    while (openList.length > 0 && iterations < maxIterations) {
      iterations++;

      // Get node with lowest f score
      openList.sort((a, b) => a.f - b.f);
      const current = openList.shift()!;
      
      const currentKey = `${current.x},${current.y}`;
      closedSet.add(currentKey);
      current.closed = true;

      // Check if we reached the goal
      const dx = current.x - goal.x;
      const dy = current.y - goal.y;
      const distToGoal = Math.sqrt(dx * dx + dy * dy);
      
      if (distToGoal < this.gridSize) {
        // Reconstruct path
        const path: Vector2D[] = [];
        let node: PathNode | null = current;
        
        while (node !== null) {
          path.unshift({ x: node.x, y: node.y });
          node = node.parent;
        }

        // Add goal if not exactly at current
        if (distToGoal > 0.001) {
          path.push({ x: goal.x, y: goal.y });
        }

        // Apply path smoothing
        const finalPath = smoothPath ? this.smoothPath(path, unitRadius) : path;

        // Cache the result
        this.pathCache.set(cacheKey, {
          path: finalPath,
          timestamp: Date.now(),
          obstacleHash
        });

        return finalPath;
      }

      // Check neighbors
      const neighbors = this.getNeighbors(current, allowDiagonal);

      for (const neighborPos of neighbors) {
        const neighborKey = `${neighborPos.x},${neighborPos.y}`;

        if (closedSet.has(neighborKey)) {
          continue;
        }

        // Check collision
        if (this.isColliding(neighborPos.x, neighborPos.y, unitRadius)) {
          continue;
        }

        // Calculate costs
        const dx = neighborPos.x - current.x;
        const dy = neighborPos.y - current.y;
        const moveCost = Math.sqrt(dx * dx + dy * dy);
        const avoidanceCost = this.getObstacleAvoidanceCost(neighborPos.x, neighborPos.y, unitRadius, avoidanceWeight);
        const tentativeG = current.g + moveCost + avoidanceCost;

        let neighbor = nodeMap.get(neighborKey);

        if (!neighbor) {
          neighbor = {
            x: neighborPos.x,
            y: neighborPos.y,
            g: Infinity,
            h: this.heuristic(neighborPos, goal, unitRadius, avoidanceWeight) * heuristicWeight,
            f: Infinity,
            parent: null,
            closed: false,
            opened: false
          };
          nodeMap.set(neighborKey, neighbor);
        }

        if (tentativeG < neighbor.g) {
          neighbor.parent = current;
          neighbor.g = tentativeG;
          neighbor.f = neighbor.g + neighbor.h;

          if (!neighbor.opened) {
            neighbor.opened = true;
            openList.push(neighbor);
          }
        }
      }
    }

    // No path found
    return null;
  }

  /**
   * Find paths for multiple units (group pathfinding)
   * Uses spatial separation to avoid unit clustering
   */
  findGroupPaths(
    units: Array<{ id: string; start: Vector2D; goal: Vector2D }>,
    options: PathfindingOptions = {}
  ): Map<string, Vector2D[] | null> {
    const results = new Map<string, Vector2D[] | null>();
    const unitRadius = options.unitRadius || 0.5;
    const separationRadius = unitRadius * 2.5;

    // Sort units by distance to goal (prioritize units farther from goal)
    const sortedUnits = [...units].sort((a, b) => {
      const distA = Math.sqrt((a.goal.x - a.start.x) ** 2 + (a.goal.y - a.start.y) ** 2);
      const distB = Math.sqrt((b.goal.x - b.start.x) ** 2 + (b.goal.y - b.start.y) ** 2);
      return distB - distA;
    });

    // Temporary obstacles for already-pathed units
    const tempObstacles = new Map<string, Obstacle>();

    for (const unit of sortedUnits) {
      // Add temporary obstacles from other units' paths
      for (const [obstId, obstacle] of tempObstacles) {
        this.obstacles.set(obstId, obstacle);
      }

      // Find path for this unit
      const path = this.findPath(unit.start, unit.goal, options);
      results.set(unit.id, path);

      // Remove temporary obstacles
      for (const obstId of tempObstacles.keys()) {
        this.obstacles.delete(obstId);
      }

      // Add waypoints as temporary obstacles for next units
      if (path && path.length > 1) {
        // Add obstacles at key waypoints to create separation
        const waypointStep = Math.max(1, Math.floor(path.length / 5));
        for (let i = 0; i < path.length; i += waypointStep) {
          const waypoint = path[i];
          const tempObstId = `temp_${unit.id}_${i}`;
          tempObstacles.set(tempObstId, {
            x: waypoint.x,
            y: waypoint.y,
            radius: separationRadius,
            isDynamic: true
          });
        }
      }
    }

    return results;
  }

  /**
   * Clear path cache (useful when map changes significantly)
   */
  clearCache(): void {
    this.pathCache.clear();
  }

  /**
   * Get statistics for debugging
   */
  getStats(): { obstacleCount: number; cacheSize: number; dynamicObstacleCount: number } {
    return {
      obstacleCount: this.obstacles.size,
      cacheSize: this.pathCache.size,
      dynamicObstacleCount: this.dynamicObstacles.size
    };
  }
}

/**
 * Create a global pathfinder instance
 */
export function createPathfinder(gridSize: number = 1.0): Pathfinder {
  return new Pathfinder(gridSize);
}
