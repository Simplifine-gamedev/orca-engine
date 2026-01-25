/**
 * Pathfinding System - A* pathfinding with support for walls and gates
 */

import { wallStore, Position, Gate } from '../store/wallStore';

interface GridNode {
  x: number;
  y: number;
  g: number; // Cost from start
  h: number; // Heuristic cost to end
  f: number; // Total cost (g + h)
  parent?: GridNode;
  walkable: boolean;
}

interface PathfindingOptions {
  unitOwnerId: string; // For checking if gates are accessible
  gridWidth: number;
  gridHeight: number;
  allowDiagonal?: boolean;
}

/**
 * Calculate path from start to end using A* algorithm
 */
export function findPath(
  start: Position,
  end: Position,
  options: PathfindingOptions
): Position[] | null {
  const { unitOwnerId, gridWidth, gridHeight, allowDiagonal = true } = options;

  // Initialize grid
  const grid = createGrid(gridWidth, gridHeight, unitOwnerId);

  // Get start and end nodes
  const startNode = grid[Math.floor(start.y)]?.[Math.floor(start.x)];
  const endNode = grid[Math.floor(end.y)]?.[Math.floor(end.x)];

  if (!startNode || !endNode || !endNode.walkable) {
    return null; // Invalid start/end positions
  }

  const openList: GridNode[] = [];
  const closedList: Set<string> = new Set();

  openList.push(startNode);

  while (openList.length > 0) {
    // Get node with lowest f cost
    let currentIndex = 0;
    for (let i = 1; i < openList.length; i++) {
      if (openList[i].f < openList[currentIndex].f) {
        currentIndex = i;
      }
    }

    const currentNode = openList[currentIndex];

    // Found the goal
    if (currentNode.x === endNode.x && currentNode.y === endNode.y) {
      return reconstructPath(currentNode);
    }

    // Move current node from open to closed list
    openList.splice(currentIndex, 1);
    closedList.add(`${currentNode.x},${currentNode.y}`);

    // Check neighbors
    const neighbors = getNeighbors(currentNode, grid, allowDiagonal);
    
    for (const neighbor of neighbors) {
      if (!neighbor.walkable || closedList.has(`${neighbor.x},${neighbor.y}`)) {
        continue;
      }

      const tentativeG = currentNode.g + calculateDistance(currentNode, neighbor);

      const existingIndex = openList.findIndex(
        (node) => node.x === neighbor.x && node.y === neighbor.y
      );

      if (existingIndex === -1) {
        // New node - add to open list
        neighbor.g = tentativeG;
        neighbor.h = heuristic(neighbor, endNode);
        neighbor.f = neighbor.g + neighbor.h;
        neighbor.parent = currentNode;
        openList.push(neighbor);
      } else if (tentativeG < openList[existingIndex].g) {
        // Better path found - update node
        openList[existingIndex].g = tentativeG;
        openList[existingIndex].f = tentativeG + openList[existingIndex].h;
        openList[existingIndex].parent = currentNode;
      }
    }
  }

  // No path found
  return null;
}

/**
 * Create grid with walls and gates marked
 */
function createGrid(
  width: number,
  height: number,
  unitOwnerId: string
): GridNode[][] {
  const grid: GridNode[][] = [];

  // Initialize empty grid
  for (let y = 0; y < height; y++) {
    grid[y] = [];
    for (let x = 0; x < width; x++) {
      grid[y][x] = {
        x,
        y,
        g: 0,
        h: 0,
        f: 0,
        walkable: true,
      };
    }
  }

  // Mark walls as unwalkable
  const walls = wallStore.getAllWalls();
  for (const wall of walls) {
    const x = Math.floor(wall.position.x);
    const y = Math.floor(wall.position.y);
    if (grid[y]?.[x]) {
      grid[y][x].walkable = false;
    }
  }

  // Mark gates based on owner and state
  const gates = wallStore.getAllGates();
  for (const gate of gates) {
    const x = Math.floor(gate.position.x);
    const y = Math.floor(gate.position.y);
    
    if (grid[y]?.[x]) {
      // Gate is walkable if:
      // 1. It's open, OR
      // 2. Unit is friendly to gate owner (will trigger auto-open)
      const isFriendly = wallStore.isFriendlyUnit(unitOwnerId, gate.ownerId);
      grid[y][x].walkable = gate.isOpen || isFriendly;
    }
  }

  return grid;
}

/**
 * Get neighboring nodes
 */
function getNeighbors(
  node: GridNode,
  grid: GridNode[][],
  allowDiagonal: boolean
): GridNode[] {
  const neighbors: GridNode[] = [];
  const { x, y } = node;

  // Cardinal directions
  const directions = [
    { dx: 0, dy: -1 }, // North
    { dx: 1, dy: 0 },  // East
    { dx: 0, dy: 1 },  // South
    { dx: -1, dy: 0 }, // West
  ];

  // Diagonal directions
  if (allowDiagonal) {
    directions.push(
      { dx: -1, dy: -1 }, // Northwest
      { dx: 1, dy: -1 },  // Northeast
      { dx: 1, dy: 1 },   // Southeast
      { dx: -1, dy: 1 }   // Southwest
    );
  }

  for (const { dx, dy } of directions) {
    const newX = x + dx;
    const newY = y + dy;

    if (grid[newY]?.[newX]) {
      // For diagonal moves, check if cardinal neighbors are walkable
      if (allowDiagonal && Math.abs(dx) === 1 && Math.abs(dy) === 1) {
        const horizontalWalkable = grid[y]?.[newX]?.walkable ?? false;
        const verticalWalkable = grid[newY]?.[x]?.walkable ?? false;
        
        // Only allow diagonal if both adjacent cardinals are walkable
        if (horizontalWalkable && verticalWalkable) {
          neighbors.push(grid[newY][newX]);
        }
      } else {
        neighbors.push(grid[newY][newX]);
      }
    }
  }

  return neighbors;
}

/**
 * Calculate distance between two nodes
 */
function calculateDistance(node1: GridNode, node2: GridNode): number {
  const dx = Math.abs(node1.x - node2.x);
  const dy = Math.abs(node1.y - node2.y);

  // Diagonal distance: D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
  // where D = 1 for cardinal move, D2 = sqrt(2) for diagonal
  if (dx > dy) {
    return 1.414 * dy + (dx - dy);
  } else {
    return 1.414 * dx + (dy - dx);
  }
}

/**
 * Heuristic function (Manhattan distance)
 */
function heuristic(node: GridNode, goal: GridNode): number {
  return Math.abs(node.x - goal.x) + Math.abs(node.y - goal.y);
}

/**
 * Reconstruct path from end node to start
 */
function reconstructPath(endNode: GridNode): Position[] {
  const path: Position[] = [];
  let current: GridNode | undefined = endNode;

  while (current) {
    path.unshift({ x: current.x, y: current.y });
    current = current.parent;
  }

  return path;
}

/**
 * Check if position is blocked by a wall or closed gate
 */
export function isPositionBlocked(
  position: Position,
  unitOwnerId: string
): boolean {
  const x = Math.floor(position.x);
  const y = Math.floor(position.y);

  // Check walls
  const walls = wallStore.getAllWalls();
  for (const wall of walls) {
    if (Math.floor(wall.position.x) === x && Math.floor(wall.position.y) === y) {
      return true; // Blocked by wall
    }
  }

  // Check gates
  const gates = wallStore.getAllGates();
  for (const gate of gates) {
    if (Math.floor(gate.position.x) === x && Math.floor(gate.position.y) === y) {
      const isFriendly = wallStore.isFriendlyUnit(unitOwnerId, gate.ownerId);
      // Blocked if gate is closed AND unit is not friendly
      return !gate.isOpen && !isFriendly;
    }
  }

  return false; // Not blocked
}

/**
 * Get gates along a path that will need to be opened
 */
export function getGatesAlongPath(
  path: Position[],
  unitOwnerId: string
): Gate[] {
  const gatesOnPath: Gate[] = [];
  const gates = wallStore.getAllGates();

  for (const position of path) {
    for (const gate of gates) {
      const gateX = Math.floor(gate.position.x);
      const gateY = Math.floor(gate.position.y);
      const pathX = Math.floor(position.x);
      const pathY = Math.floor(position.y);

      if (gateX === pathX && gateY === pathY) {
        const isFriendly = wallStore.isFriendlyUnit(unitOwnerId, gate.ownerId);
        if (isFriendly) {
          gatesOnPath.push(gate);
        }
      }
    }
  }

  return gatesOnPath;
}

/**
 * Smooth path by removing unnecessary waypoints
 */
export function smoothPath(path: Position[]): Position[] {
  if (path.length <= 2) {
    return path;
  }

  const smoothed: Position[] = [path[0]];
  let currentIndex = 0;

  while (currentIndex < path.length - 1) {
    // Try to skip as many waypoints as possible
    let farthestIndex = currentIndex + 1;
    
    for (let i = path.length - 1; i > currentIndex + 1; i--) {
      if (hasLineOfSight(path[currentIndex], path[i])) {
        farthestIndex = i;
        break;
      }
    }

    smoothed.push(path[farthestIndex]);
    currentIndex = farthestIndex;
  }

  return smoothed;
}

/**
 * Check if there's a line of sight between two positions
 */
function hasLineOfSight(start: Position, end: Position): boolean {
  // Simple line of sight check using Bresenham's line algorithm
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);
  const sx = start.x < end.x ? 1 : -1;
  const sy = start.y < end.y ? 1 : -1;
  let err = dx - dy;

  let x = Math.floor(start.x);
  let y = Math.floor(start.y);
  const endX = Math.floor(end.x);
  const endY = Math.floor(end.y);

  while (x !== endX || y !== endY) {
    // Check if current position has a wall (not checking gates for LOS)
    const walls = wallStore.getAllWalls();
    for (const wall of walls) {
      if (Math.floor(wall.position.x) === x && Math.floor(wall.position.y) === y) {
        return false;
      }
    }

    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }

  return true;
}
