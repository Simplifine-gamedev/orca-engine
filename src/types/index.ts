export interface Resource {
  id: string;
  type: 'goldmine' | 'tree';
  position: { x: number; y: number };
  amountRemaining: number;
  maxAmount: number;
  workersAssigned: number;
  gatherRate: number;
}

export interface GameState {
  resources: Resource[];
  selectedResourceId: string | null;
  selectResource: (id: string | null) => void;
  addWorkerToResource: (id: string) => void;
  removeWorkerFromResource: (id: string) => void;
}
