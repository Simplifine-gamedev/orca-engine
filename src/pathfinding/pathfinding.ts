// A* Pathfinding algorithm with gate awareness
import { Position, PathNode, Unit } from '../types';
import { wallStore } from '../store/wallStore';

export class Pathfinder {
  private gridWidth: number;
  private gridHeight: number;

  constructor(gridWidth: number, gridHeight: number) {
    this.gridWidth = gridWidth;
    this.gridHeight = gridHeight;
  }

  // Calculate heuristic (Manhattan distance)
  private heuristic(pos1: Position, pos2: Position): number {
    return Math.abs(pos1.x - pos2.x) + Math.abs(pos1.y - pos2.y);
  }

  // Get neighboring positions (4-directional)
  private getNeighbors(position: Position): Position[] {
    const neighbors: Position[] = [];
    const directions = [
      { x: 0, y: -1 }, // up
      { x: 1, y: 0 },  // right
      { x: 0, y: 1 },  // down
      { x: -1, y: 0 }, // left
    ];

    for (const dir of directions) {
      const newPos: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y,
      };

      // Check if position is within grid bounds
      if (
        newPos.x >= 0 &&
        newPos.x < this.gridWidth &&
        newPos.y >= 0 &&
        newPos.y < this.gridHeight
      ) {
        neighbors.push(newPos);
      }
    }

    return neighbors;
  }

  // Check if two positions are equal
  private positionsEqual(pos1: Position, pos2: Position): boolean {
    return pos1.x === pos2.x && pos1.y === pos2.y;
  }

  // Find a node in a list by position
  private findNodeByPosition(nodes: PathNode[], position: Position): PathNode | undefined {
    return nodes.find(node => this.positionsEqual(node.position, position));
  }

  // Reconstruct path from goal to start
  private reconstructPath(node: PathNode): Position[] {
    const path: Position[] = [];
    let current: PathNode | undefined = node;

    while (current) {
      path.unshift(current.position);
      current = current.parent;
    }

    return path;
  }

  // A* pathfinding algorithm
  findPath(start: Position, goal: Position, team: 'friendly' | 'enemy'): Position[] | null {
    // Early exit if start or goal is blocked
    if (wallStore.isPositionBlocked(start, team) || wallStore.isPositionBlocked(goal, team)) {
      return null;
    }

    const openList: PathNode[] = [];
    const closedList: PathNode[] = [];

    // Create start node
    const startNode: PathNode = {
      position: start,
      g: 0,
      h: this.heuristic(start, goal),
      f: 0,
    };
    startNode.f = startNode.g + startNode.h;
    openList.push(startNode);

    while (openList.length > 0) {
      // Find node with lowest f score
      let currentIndex = 0;
      for (let i = 1; i < openList.length; i++) {
        if (openList[i].f < openList[currentIndex].f) {
          currentIndex = i;
        }
      }

      const current = openList[currentIndex];

      // Check if we reached the goal
      if (this.positionsEqual(current.position, goal)) {
        return this.reconstructPath(current);
      }

      // Move current from open to closed list
      openList.splice(currentIndex, 1);
      closedList.push(current);

      // Check all neighbors
      const neighbors = this.getNeighbors(current.position);

      for (const neighborPos of neighbors) {
        // Skip if neighbor is blocked
        if (wallStore.isPositionBlocked(neighborPos, team)) {
          continue;
        }

        // Skip if neighbor is in closed list
        if (this.findNodeByPosition(closedList, neighborPos)) {
          continue;
        }

        // Calculate g score
        const g = current.g + 1;

        // Check if neighbor is in open list
        const existingNode = this.findNodeByPosition(openList, neighborPos);

        if (!existingNode) {
          // Add new node to open list
          const h = this.heuristic(neighborPos, goal);
          const newNode: PathNode = {
            position: neighborPos,
            g,
            h,
            f: g + h,
            parent: current,
          };
          openList.push(newNode);
        } else if (g < existingNode.g) {
          // Update existing node if this path is better
          existingNode.g = g;
          existingNode.f = g + existingNode.h;
          existingNode.parent = current;
        }
      }
    }

    // No path found
    return null;
  }

  // Update grid dimensions if needed
  setGridSize(width: number, height: number): void {
    this.gridWidth = width;
    this.gridHeight = height;
  }

  // Check if a path exists between two points
  hasPath(start: Position, goal: Position, team: 'friendly' | 'enemy'): boolean {
    return this.findPath(start, goal, team) !== null;
  }

  // Get the next position in the path (for unit movement)
  getNextPosition(currentPos: Position, goal: Position, team: 'friendly' | 'enemy'): Position | null {
    const path = this.findPath(currentPos, goal, team);
    if (!path || path.length < 2) {
      return null;
    }
    // Return the second position in the path (first is current position)
    return path[1];
  }
}

// Create and export a default pathfinder instance
export const pathfinder = new Pathfinder(50, 50); // Default 50x50 grid
export default pathfinder;
