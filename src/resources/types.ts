/**
 * Core types for game resources
 */

export interface Position {
  x: number;
  y: number;
}

export interface Resource {
  id: string;
  type: ResourceType;
  position: Position;
  isSelected?: boolean;
}

export enum ResourceType {
  GOLD_MINE = 'gold_mine',
  STONE_QUARRY = 'stone_quarry',
  WOOD_FOREST = 'wood_forest',
}

export interface GoldMineState extends Resource {
  type: ResourceType.GOLD_MINE;
  goldRemaining: number;
  maxGold: number;
  harvestRate: number;
}

export interface ResourceDepletion {
  current: number;
  max: number;
  percentage: number;
  isLow: boolean;
  isDepleted: boolean;
}

/**
 * Calculate depletion status for a resource
 */
export function getDepletionStatus(
  current: number,
  max: number
): ResourceDepletion {
  const percentage = (current / max) * 100;
  return {
    current,
    max,
    percentage,
    isLow: percentage < 25,
    isDepleted: current <= 0,
  };
}
