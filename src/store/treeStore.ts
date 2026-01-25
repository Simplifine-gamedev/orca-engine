import { create } from 'zustand';
import { Tree, Position, TREE_REGROWTH_TIME, TREE_REGROWTH_RATE } from '../types';

interface TreeStore {
  trees: Tree[];
  
  // Tree management
  addTree: (position: Position, woodAmount?: number) => void;
  removeTree: (treeId: string) => void;
  harvestTree: (treeId: string, amount: number) => number;
  updateTree: (treeId: string, updates: Partial<Tree>) => void;
  
  // Tree generation
  generateTrees: (count: number, mapWidth: number, mapHeight: number) => void;
  
  // Tree regrowth
  updateTreeGrowth: (deltaTime: number) => void;
  
  // Queries
  findNearestTree: (position: Position, minWood?: number) => Tree | null;
  getTreeById: (treeId: string) => Tree | null;
}

const generateTreeId = (): string => {
  return `tree_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

const calculateDistance = (pos1: Position, pos2: Position): number => {
  return Math.sqrt(Math.pow(pos2.x - pos1.x, 2) + Math.pow(pos2.y - pos1.y, 2));
};

export const useTreeStore = create<TreeStore>((set, get) => ({
  trees: [],

  addTree: (position: Position, woodAmount: number = 100) => {
    const newTree: Tree = {
      id: generateTreeId(),
      position,
      woodAmount,
      maxWood: 100,
      isGrowing: false,
      lastHarvestTime: 0,
      isDepleted: false,
    };
    
    set((state) => ({
      trees: [...state.trees, newTree],
    }));
  },

  removeTree: (treeId: string) => {
    set((state) => ({
      trees: state.trees.filter((tree) => tree.id !== treeId),
    }));
  },

  harvestTree: (treeId: string, amount: number): number => {
    let harvestedAmount = 0;
    
    set((state) => ({
      trees: state.trees.map((tree) => {
        if (tree.id === treeId && !tree.isDepleted) {
          const actualHarvest = Math.min(amount, tree.woodAmount);
          harvestedAmount = actualHarvest;
          
          const newWoodAmount = tree.woodAmount - actualHarvest;
          const isDepleted = newWoodAmount <= 0;
          
          return {
            ...tree,
            woodAmount: newWoodAmount,
            isDepleted,
            isGrowing: isDepleted ? false : true,
            lastHarvestTime: Date.now(),
          };
        }
        return tree;
      }),
    }));
    
    return harvestedAmount;
  },

  updateTree: (treeId: string, updates: Partial<Tree>) => {
    set((state) => ({
      trees: state.trees.map((tree) =>
        tree.id === treeId ? { ...tree, ...updates } : tree
      ),
    }));
  },

  generateTrees: (count: number, mapWidth: number, mapHeight: number) => {
    const newTrees: Tree[] = [];
    
    for (let i = 0; i < count; i++) {
      const tree: Tree = {
        id: generateTreeId(),
        position: {
          x: Math.random() * mapWidth,
          y: Math.random() * mapHeight,
        },
        woodAmount: 80 + Math.random() * 40, // 80-120 wood per tree
        maxWood: 100,
        isGrowing: false,
        lastHarvestTime: 0,
        isDepleted: false,
      };
      newTrees.push(tree);
    }
    
    set((state) => ({
      trees: [...state.trees, ...newTrees],
    }));
  },

  updateTreeGrowth: (deltaTime: number) => {
    const now = Date.now();
    
    set((state) => ({
      trees: state.trees.map((tree) => {
        if (!tree.isGrowing || tree.woodAmount >= tree.maxWood) {
          return tree;
        }
        
        // Check if enough time has passed since last harvest
        const timeSinceHarvest = now - tree.lastHarvestTime;
        
        if (timeSinceHarvest < TREE_REGROWTH_TIME) {
          // Still in cooldown period
          return tree;
        }
        
        // Regrow wood
        const growthAmount = TREE_REGROWTH_RATE * (deltaTime / 1000);
        const newWoodAmount = Math.min(tree.woodAmount + growthAmount, tree.maxWood);
        
        return {
          ...tree,
          woodAmount: newWoodAmount,
          isGrowing: newWoodAmount < tree.maxWood,
        };
      }),
    }));
  },

  findNearestTree: (position: Position, minWood: number = 10): Tree | null => {
    const trees = get().trees;
    const availableTrees = trees.filter(
      (tree) => !tree.isDepleted && tree.woodAmount >= minWood
    );
    
    if (availableTrees.length === 0) {
      return null;
    }
    
    let nearestTree: Tree | null = null;
    let minDistance = Infinity;
    
    availableTrees.forEach((tree) => {
      const distance = calculateDistance(position, tree.position);
      if (distance < minDistance) {
        minDistance = distance;
        nearestTree = tree;
      }
    });
    
    return nearestTree;
  },

  getTreeById: (treeId: string): Tree | null => {
    return get().trees.find((tree) => tree.id === treeId) || null;
  },
}));
