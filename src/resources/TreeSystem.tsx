import React, { useEffect, useRef, useState } from 'react';
import { useTreeStore } from '../store/treeStore';
import { useGameStore } from '../store/gameStore';
import { Tree, Worker, Position, WOOD_PER_HARVEST, WORKER_CARRY_CAPACITY } from '../types';

interface TreeSystemProps {
  mapWidth: number;
  mapHeight: number;
  treeCount?: number;
}

const TreeSystem: React.FC<TreeSystemProps> = ({ 
  mapWidth, 
  mapHeight, 
  treeCount = 50 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedWorker, setSelectedWorker] = useState<string | null>(null);
  
  const trees = useTreeStore((state) => state.trees);
  const generateTrees = useTreeStore((state) => state.generateTrees);
  const harvestTree = useTreeStore((state) => state.harvestTree);
  const updateTreeGrowth = useTreeStore((state) => state.updateTreeGrowth);
  const findNearestTree = useTreeStore((state) => state.findNearestTree);
  
  const workers = useGameStore((state) => state.workers);
  const addWood = useGameStore((state) => state.addWood);
  const updateWorker = useGameStore((state) => state.updateWorker);
  const isPaused = useGameStore((state) => state.isPaused);

  // Initialize trees on mount
  useEffect(() => {
    if (trees.length === 0) {
      generateTrees(treeCount, mapWidth, mapHeight);
    }
  }, []);

  // Game loop for tree growth and worker actions
  useEffect(() => {
    if (isPaused) return;

    const interval = setInterval(() => {
      // Update tree growth
      updateTreeGrowth(1000); // 1 second tick

      // Process worker actions
      workers.forEach((worker) => {
        if (worker.isGathering && worker.targetTreeId) {
          const tree = trees.find((t) => t.id === worker.targetTreeId);
          
          if (tree && !tree.isDepleted) {
            // Worker is at tree, harvest wood
            if (worker.carryingWood < worker.maxCarryCapacity) {
              const harvestAmount = Math.min(
                WOOD_PER_HARVEST,
                worker.maxCarryCapacity - worker.carryingWood
              );
              
              const actualHarvested = harvestTree(tree.id, harvestAmount);
              
              updateWorker(worker.id, {
                carryingWood: worker.carryingWood + actualHarvested,
              });
            } else {
              // Worker is full, return to deposit
              updateWorker(worker.id, {
                isGathering: false,
                targetTreeId: null,
              });
              
              // Deposit wood
              addWood(worker.carryingWood);
              updateWorker(worker.id, {
                carryingWood: 0,
              });
            }
          } else {
            // Tree depleted, find another
            const nextTree = findNearestTree(worker.position);
            if (nextTree) {
              updateWorker(worker.id, {
                targetTreeId: nextTree.id,
                position: nextTree.position,
              });
            } else {
              // No trees available
              updateWorker(worker.id, {
                isGathering: false,
                targetTreeId: null,
              });
            }
          }
        }
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isPaused, workers, trees]);

  // Render trees and workers
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, mapWidth, mapHeight);

    // Draw trees
    trees.forEach((tree) => {
      const alpha = tree.isDepleted ? 0.3 : 0.8;
      const size = 20 + (tree.woodAmount / tree.maxWood) * 10;
      
      ctx.save();
      ctx.globalAlpha = alpha;
      
      // Tree trunk
      ctx.fillStyle = '#8B4513';
      ctx.fillRect(tree.position.x - 3, tree.position.y, 6, 15);
      
      // Tree foliage
      ctx.fillStyle = tree.isGrowing ? '#90EE90' : '#228B22';
      ctx.beginPath();
      ctx.arc(tree.position.x, tree.position.y - 5, size / 2, 0, Math.PI * 2);
      ctx.fill();
      
      // Wood amount indicator
      if (tree.woodAmount > 0) {
        ctx.fillStyle = '#FFF';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(Math.floor(tree.woodAmount).toString(), tree.position.x, tree.position.y - size);
      }
      
      ctx.restore();
    });

    // Draw workers
    workers.forEach((worker) => {
      ctx.fillStyle = worker.isGathering ? '#FFD700' : '#4169E1';
      ctx.beginPath();
      ctx.arc(worker.position.x, worker.position.y, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw selection indicator
      if (worker.id === selectedWorker) {
        ctx.strokeStyle = '#FFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(worker.position.x, worker.position.y, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      // Draw carrying indicator
      if (worker.carryingWood > 0) {
        ctx.fillStyle = '#8B4513';
        ctx.fillRect(worker.position.x - 5, worker.position.y - 15, 10, 5);
        ctx.fillStyle = '#FFF';
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(worker.carryingWood.toString(), worker.position.x, worker.position.y - 18);
      }
    });
  }, [trees, workers, selectedWorker]);

  // Handle canvas click to select workers or assign tasks
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Check if clicked on a worker
    const clickedWorker = workers.find((worker) => {
      const distance = Math.sqrt(
        Math.pow(worker.position.x - x, 2) + Math.pow(worker.position.y - y, 2)
      );
      return distance < 10;
    });

    if (clickedWorker) {
      setSelectedWorker(clickedWorker.id);
      return;
    }

    // If worker selected, check if clicked on a tree to assign gathering
    if (selectedWorker) {
      const clickedTree = trees.find((tree) => {
        const distance = Math.sqrt(
          Math.pow(tree.position.x - x, 2) + Math.pow(tree.position.y - y, 2)
        );
        return distance < 20 && !tree.isDepleted;
      });

      if (clickedTree) {
        updateWorker(selectedWorker, {
          isGathering: true,
          targetTreeId: clickedTree.id,
          position: clickedTree.position,
        });
      }
    }
  };

  return (
    <div className="tree-system">
      <canvas
        ref={canvasRef}
        width={mapWidth}
        height={mapHeight}
        onClick={handleCanvasClick}
        style={{
          border: '2px solid #333',
          backgroundColor: '#87CEEB',
          cursor: 'pointer',
        }}
      />
      <div className="tree-stats" style={{ marginTop: '10px', fontSize: '14px' }}>
        <div>Total Trees: {trees.length}</div>
        <div>Active Trees: {trees.filter((t) => !t.isDepleted).length}</div>
        <div>Growing Trees: {trees.filter((t) => t.isGrowing).length}</div>
        <div>Depleted Trees: {trees.filter((t) => t.isDepleted).length}</div>
        <div>Workers Gathering: {workers.filter((w) => w.isGathering).length}</div>
      </div>
    </div>
  );
};

export default TreeSystem;
