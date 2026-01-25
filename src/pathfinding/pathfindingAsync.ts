/**
 * Asynchronous Pathfinding for Orca RTS
 * 
 * Features:
 * - Non-blocking pathfinding using Web Workers or async batching
 * - Request queuing and prioritization
 * - Progressive pathfinding for large groups
 * - Parallel path computation
 */

import {
  Vector2D,
  Obstacle,
  PathfindingOptions,
  Pathfinder
} from './pathfinding';

export interface PathRequest {
  id: string;
  start: Vector2D;
  goal: Vector2D;
  options?: PathfindingOptions;
  priority?: number; // Higher = more urgent
  timestamp: number;
}

export interface PathResult {
  id: string;
  path: Vector2D[] | null;
  success: boolean;
  computeTime: number;
}

export interface GroupPathRequest {
  units: Array<{ id: string; start: Vector2D; goal: Vector2D }>;
  options?: PathfindingOptions;
  priority?: number;
}

/**
 * Async Pathfinding Manager
 * Handles pathfinding requests asynchronously with queuing and batching
 */
export class AsyncPathfinder {
  private pathfinder: Pathfinder;
  private requestQueue: PathRequest[];
  private groupQueue: GroupPathRequest[];
  private processing: boolean;
  private maxBatchSize: number;
  private maxProcessingTime: number; // Max time per frame in ms
  private callbacks: Map<string, (result: PathResult) => void>;
  private groupCallbacks: Map<string, (results: Map<string, Vector2D[] | null>) => void>;

  constructor(pathfinder: Pathfinder, maxBatchSize: number = 10, maxProcessingTime: number = 16) {
    this.pathfinder = pathfinder;
    this.requestQueue = [];
    this.groupQueue = [];
    this.processing = false;
    this.maxBatchSize = maxBatchSize;
    this.maxProcessingTime = maxProcessingTime;
    this.callbacks = new Map();
    this.groupCallbacks = new Map();
  }

  /**
   * Request a path asynchronously
   */
  requestPath(
    id: string,
    start: Vector2D,
    goal: Vector2D,
    options?: PathfindingOptions,
    priority: number = 0
  ): Promise<Vector2D[] | null> {
    return new Promise((resolve) => {
      const request: PathRequest = {
        id,
        start,
        goal,
        options,
        priority,
        timestamp: Date.now()
      };

      this.requestQueue.push(request);
      this.callbacks.set(id, (result: PathResult) => {
        resolve(result.path);
      });

      // Sort queue by priority (higher priority first)
      this.requestQueue.sort((a, b) => {
        if (b.priority !== a.priority) {
          return (b.priority || 0) - (a.priority || 0);
        }
        return a.timestamp - b.timestamp; // FIFO for same priority
      });

      // Start processing if not already running
      if (!this.processing) {
        this.processQueue();
      }
    });
  }

  /**
   * Request paths for a group of units
   */
  requestGroupPaths(
    units: Array<{ id: string; start: Vector2D; goal: Vector2D }>,
    options?: PathfindingOptions,
    priority: number = 0
  ): Promise<Map<string, Vector2D[] | null>> {
    return new Promise((resolve) => {
      const groupId = `group_${Date.now()}_${Math.random()}`;
      
      const request: GroupPathRequest = {
        units,
        options,
        priority
      };

      this.groupQueue.push(request);
      this.groupCallbacks.set(groupId, (results: Map<string, Vector2D[] | null>) => {
        resolve(results);
      });

      // Start processing if not already running
      if (!this.processing) {
        this.processQueue();
      }
    });
  }

  /**
   * Cancel a pathfinding request
   */
  cancelRequest(id: string): void {
    const index = this.requestQueue.findIndex(req => req.id === id);
    if (index !== -1) {
      this.requestQueue.splice(index, 1);
      this.callbacks.delete(id);
    }
  }

  /**
   * Process the request queue with time-slicing
   */
  private async processQueue(): Promise<void> {
    if (this.processing) {
      return;
    }

    this.processing = true;

    while (this.requestQueue.length > 0 || this.groupQueue.length > 0) {
      const frameStartTime = performance.now();
      let processed = 0;

      // Process group requests first (they're usually more important)
      while (this.groupQueue.length > 0 && processed < this.maxBatchSize) {
        const groupRequest = this.groupQueue.shift()!;
        
        const startTime = performance.now();
        const results = this.pathfinder.findGroupPaths(groupRequest.units, groupRequest.options);
        const computeTime = performance.now() - startTime;

        // Find and call the callback
        for (const [groupId, callback] of this.groupCallbacks.entries()) {
          callback(results);
          this.groupCallbacks.delete(groupId);
          break; // Only call the first matching callback
        }

        processed++;

        // Check if we've exceeded our time budget
        if (performance.now() - frameStartTime > this.maxProcessingTime) {
          break;
        }
      }

      // Process individual requests
      while (this.requestQueue.length > 0 && processed < this.maxBatchSize) {
        const request = this.requestQueue.shift()!;
        
        const startTime = performance.now();
        const path = this.pathfinder.findPath(request.start, request.goal, request.options);
        const computeTime = performance.now() - startTime;

        const result: PathResult = {
          id: request.id,
          path,
          success: path !== null,
          computeTime
        };

        const callback = this.callbacks.get(request.id);
        if (callback) {
          callback(result);
          this.callbacks.delete(request.id);
        }

        processed++;

        // Check if we've exceeded our time budget
        if (performance.now() - frameStartTime > this.maxProcessingTime) {
          break;
        }
      }

      // Yield to the event loop if there are more requests
      if (this.requestQueue.length > 0 || this.groupQueue.length > 0) {
        await this.sleep(0);
      }
    }

    this.processing = false;
  }

  /**
   * Sleep for a specified duration (yields to event loop)
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Update obstacle in pathfinder
   */
  addObstacle(id: string, obstacle: Obstacle): void {
    this.pathfinder.addObstacle(id, obstacle);
  }

  /**
   * Remove obstacle from pathfinder
   */
  removeObstacle(id: string): void {
    this.pathfinder.removeObstacle(id);
  }

  /**
   * Update obstacle position
   */
  updateObstacle(id: string, position: Vector2D, constructionState?: string): void {
    this.pathfinder.updateObstacle(id, position, constructionState);
  }

  /**
   * Clear all caches
   */
  clearCache(): void {
    this.pathfinder.clearCache();
  }

  /**
   * Get queue statistics
   */
  getStats(): {
    queueSize: number;
    groupQueueSize: number;
    processing: boolean;
    pathfinderStats: ReturnType<Pathfinder['getStats']>;
  } {
    return {
      queueSize: this.requestQueue.length,
      groupQueueSize: this.groupQueue.length,
      processing: this.processing,
      pathfinderStats: this.pathfinder.getStats()
    };
  }

  /**
   * Set processing parameters
   */
  setProcessingParams(maxBatchSize?: number, maxProcessingTime?: number): void {
    if (maxBatchSize !== undefined) {
      this.maxBatchSize = maxBatchSize;
    }
    if (maxProcessingTime !== undefined) {
      this.maxProcessingTime = maxProcessingTime;
    }
  }
}

/**
 * Progressive Pathfinding - for very large groups
 * Computes paths over multiple frames with progress callbacks
 */
export class ProgressivePathfinder {
  private asyncPathfinder: AsyncPathfinder;
  private currentBatch: Array<{ id: string; start: Vector2D; goal: Vector2D }> | null;
  private progressCallback: ((progress: number, completed: number, total: number) => void) | null;

  constructor(asyncPathfinder: AsyncPathfinder) {
    this.asyncPathfinder = asyncPathfinder;
    this.currentBatch = null;
    this.progressCallback = null;
  }

  /**
   * Find paths for a large group progressively
   */
  async findGroupPathsProgressive(
    units: Array<{ id: string; start: Vector2D; goal: Vector2D }>,
    options?: PathfindingOptions,
    onProgress?: (progress: number, completed: number, total: number) => void
  ): Promise<Map<string, Vector2D[] | null>> {
    this.currentBatch = units;
    this.progressCallback = onProgress || null;

    const results = new Map<string, Vector2D[] | null>();
    const batchSize = 5; // Process 5 units at a time
    const totalUnits = units.length;

    for (let i = 0; i < units.length; i += batchSize) {
      const batch = units.slice(i, Math.min(i + batchSize, units.length));
      
      // Request paths for this batch
      const batchPromises = batch.map(unit =>
        this.asyncPathfinder.requestPath(unit.id, unit.start, unit.goal, options, 1)
      );

      // Wait for batch to complete
      const batchResults = await Promise.all(batchPromises);

      // Store results
      batch.forEach((unit, index) => {
        results.set(unit.id, batchResults[index]);
      });

      // Report progress
      const completed = Math.min(i + batchSize, totalUnits);
      const progress = completed / totalUnits;
      
      if (this.progressCallback) {
        this.progressCallback(progress, completed, totalUnits);
      }
    }

    this.currentBatch = null;
    this.progressCallback = null;

    return results;
  }

  /**
   * Cancel current progressive pathfinding
   */
  cancel(): void {
    if (this.currentBatch) {
      for (const unit of this.currentBatch) {
        this.asyncPathfinder.cancelRequest(unit.id);
      }
      this.currentBatch = null;
      this.progressCallback = null;
    }
  }
}

/**
 * Flow Field Pathfinding - for very large unit groups moving to same goal
 * More efficient than individual A* for large groups
 */
export class FlowFieldPathfinder {
  private pathfinder: Pathfinder;
  private flowFields: Map<string, Map<string, Vector2D>>;

  constructor(pathfinder: Pathfinder) {
    this.pathfinder = pathfinder;
    this.flowFields = new Map();
  }

  /**
   * Generate a flow field for a goal position
   */
  generateFlowField(
    goal: Vector2D,
    bounds: { minX: number; maxX: number; minY: number; maxY: number },
    gridSize: number = 1.0,
    unitRadius: number = 0.5
  ): Map<string, Vector2D> {
    const flowField = new Map<string, Vector2D>();
    const integrationField = new Map<string, number>();

    // Initialize integration field with Dijkstra-like wave from goal
    const queue: Array<{ pos: Vector2D; cost: number }> = [{ pos: goal, cost: 0 }];
    const visited = new Set<string>();

    while (queue.length > 0) {
      queue.sort((a, b) => a.cost - b.cost);
      const current = queue.shift()!;
      const key = `${current.pos.x},${current.pos.y}`;

      if (visited.has(key)) {
        continue;
      }
      visited.add(key);
      integrationField.set(key, current.cost);

      // Check neighbors
      const neighbors = this.getFlowNeighbors(current.pos, gridSize, bounds);
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x},${neighbor.y}`;
        
        if (visited.has(neighborKey)) {
          continue;
        }

        // Check if neighbor is walkable
        if (!this.isWalkable(neighbor.x, neighbor.y, unitRadius)) {
          continue;
        }

        const dx = neighbor.x - current.pos.x;
        const dy = neighbor.y - current.pos.y;
        const moveCost = Math.sqrt(dx * dx + dy * dy);
        const newCost = current.cost + moveCost;

        queue.push({ pos: neighbor, cost: newCost });
      }
    }

    // Generate flow field from integration field
    for (let x = bounds.minX; x <= bounds.maxX; x += gridSize) {
      for (let y = bounds.minY; y <= bounds.maxY; y += gridSize) {
        const key = `${x},${y}`;
        const currentCost = integrationField.get(key);

        if (currentCost === undefined) {
          continue;
        }

        // Find neighbor with lowest cost
        let bestNeighbor: Vector2D | null = null;
        let lowestCost = currentCost;

        const neighbors = this.getFlowNeighbors({ x, y }, gridSize, bounds);
        for (const neighbor of neighbors) {
          const neighborKey = `${neighbor.x},${neighbor.y}`;
          const neighborCost = integrationField.get(neighborKey);

          if (neighborCost !== undefined && neighborCost < lowestCost) {
            lowestCost = neighborCost;
            bestNeighbor = neighbor;
          }
        }

        if (bestNeighbor) {
          const dx = bestNeighbor.x - x;
          const dy = bestNeighbor.y - y;
          const length = Math.sqrt(dx * dx + dy * dy);
          
          flowField.set(key, {
            x: dx / length,
            y: dy / length
          });
        }
      }
    }

    return flowField;
  }

  /**
   * Get flow direction at a position
   */
  getFlowDirection(flowField: Map<string, Vector2D>, pos: Vector2D, gridSize: number = 1.0): Vector2D | null {
    // Snap to grid
    const gridX = Math.round(pos.x / gridSize) * gridSize;
    const gridY = Math.round(pos.y / gridSize) * gridSize;
    const key = `${gridX},${gridY}`;

    return flowField.get(key) || null;
  }

  /**
   * Get neighbors for flow field generation
   */
  private getFlowNeighbors(pos: Vector2D, gridSize: number, bounds: { minX: number; maxX: number; minY: number; maxY: number }): Vector2D[] {
    const neighbors: Vector2D[] = [];
    const offsets = [
      { x: gridSize, y: 0 },
      { x: -gridSize, y: 0 },
      { x: 0, y: gridSize },
      { x: 0, y: -gridSize },
      { x: gridSize, y: gridSize },
      { x: gridSize, y: -gridSize },
      { x: -gridSize, y: gridSize },
      { x: -gridSize, y: -gridSize }
    ];

    for (const offset of offsets) {
      const nx = pos.x + offset.x;
      const ny = pos.y + offset.y;

      if (nx >= bounds.minX && nx <= bounds.maxX && ny >= bounds.minY && ny <= bounds.maxY) {
        neighbors.push({ x: nx, y: ny });
      }
    }

    return neighbors;
  }

  /**
   * Check if position is walkable
   */
  private isWalkable(x: number, y: number, unitRadius: number): boolean {
    // Access the private method through any cast
    // In production, this should be exposed properly
    return !(this.pathfinder as any).isColliding(x, y, unitRadius);
  }
}

/**
 * Create async pathfinder with default settings
 */
export function createAsyncPathfinder(pathfinder: Pathfinder): AsyncPathfinder {
  return new AsyncPathfinder(pathfinder);
}

/**
 * Create progressive pathfinder
 */
export function createProgressivePathfinder(asyncPathfinder: AsyncPathfinder): ProgressivePathfinder {
  return new ProgressivePathfinder(asyncPathfinder);
}

/**
 * Create flow field pathfinder
 */
export function createFlowFieldPathfinder(pathfinder: Pathfinder): FlowFieldPathfinder {
  return new FlowFieldPathfinder(pathfinder);
}
