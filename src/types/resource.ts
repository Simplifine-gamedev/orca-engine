export type ResourceType = 'gold_mine' | 'tree' | 'stone_quarry';

export interface ResourceData {
  id: string;
  type: ResourceType;
  name: string;
  amountRemaining: number;
  maxAmount: number;
  workersAssigned: number;
  maxWorkers: number;
  gatherRate: number; // resources per second per worker
  position: {
    x: number;
    y: number;
  };
}

export interface SelectableEntity {
  id: string;
  type: 'resource' | 'unit' | 'building';
  data: ResourceData | any;
}
