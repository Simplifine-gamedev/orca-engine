'use client'

import { useEffect } from 'react'
import { useWorkerStore, TOWN_HALL_POSITION } from '../store/workerStore'
import { useTreeStore } from '../store/treeStore'
import { useGameStore } from '../store/gameStore'
import { Position } from '../types'

const CHOP_DISTANCE = 30
const DEPOSIT_DISTANCE = 40

function distance(a: Position, b: Position): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function moveTowards(from: Position, to: Position, speed: number): Position {
  const dist = distance(from, to)
  if (dist <= speed) return to
  
  const ratio = speed / dist
  return {
    x: from.x + (to.x - from.x) * ratio,
    y: from.y + (to.y - from.y) * ratio
  }
}

export default function WorkerAI() {
  const workers = useWorkerStore(state => state.workers)
  const updateWorker = useWorkerStore(state => state.updateWorker)
  const moveWorker = useWorkerStore(state => state.moveWorker)
  
  const trees = useTreeStore(state => state.trees)
  const chopTree = useTreeStore(state => state.chopTree)
  const setTreeBeingChopped = useTreeStore(state => state.setTreeBeingChopped)
  const getAvailableTree = useTreeStore(state => state.getAvailableTree)
  
  const addResource = useGameStore(state => state.addResource)

  useEffect(() => {
    const interval = setInterval(() => {
      workers.forEach(worker => {
        switch (worker.state) {
          case 'idle': {
            // Find a tree to chop
            const tree = getAvailableTree(worker.position.x, worker.position.y)
            if (tree) {
              updateWorker(worker.id, {
                targetTreeId: tree.id,
                state: 'moving_to_tree'
              })
              setTreeBeingChopped(tree.id, true)
            }
            break
          }

          case 'moving_to_tree': {
            const tree = trees.find(t => t.id === worker.targetTreeId)
            if (!tree || tree.wood <= 0) {
              // Tree is gone or depleted
              if (tree) setTreeBeingChopped(tree.id, false)
              updateWorker(worker.id, {
                targetTreeId: null,
                state: 'idle'
              })
              break
            }

            const dist = distance(worker.position, tree.position)
            if (dist <= CHOP_DISTANCE) {
              updateWorker(worker.id, { state: 'chopping' })
            } else {
              const newPos = moveTowards(worker.position, tree.position, worker.moveSpeed)
              moveWorker(worker.id, newPos)
            }
            break
          }

          case 'chopping': {
            const tree = trees.find(t => t.id === worker.targetTreeId)
            if (!tree || tree.wood <= 0) {
              // Tree depleted
              if (tree) setTreeBeingChopped(tree.id, false)
              
              if (worker.carryingWood > 0) {
                updateWorker(worker.id, {
                  targetTreeId: null,
                  state: 'returning'
                })
              } else {
                updateWorker(worker.id, {
                  targetTreeId: null,
                  state: 'idle'
                })
              }
              break
            }

            // Chop wood
            const chopAmount = Math.min(worker.chopSpeed, worker.maxCarryCapacity - worker.carryingWood)
            const actualChop = chopTree(tree.id, chopAmount)
            
            const newCarrying = worker.carryingWood + actualChop
            
            if (newCarrying >= worker.maxCarryCapacity || tree.wood <= actualChop) {
              setTreeBeingChopped(tree.id, false)
              updateWorker(worker.id, {
                carryingWood: newCarrying,
                targetTreeId: null,
                state: 'returning'
              })
            } else {
              updateWorker(worker.id, {
                carryingWood: newCarrying
              })
            }
            break
          }

          case 'returning': {
            const dist = distance(worker.position, TOWN_HALL_POSITION)
            if (dist <= DEPOSIT_DISTANCE) {
              updateWorker(worker.id, { state: 'depositing' })
            } else {
              const newPos = moveTowards(worker.position, TOWN_HALL_POSITION, worker.moveSpeed)
              moveWorker(worker.id, newPos)
            }
            break
          }

          case 'depositing': {
            // Deposit wood
            addResource('wood', worker.carryingWood)
            updateWorker(worker.id, {
              carryingWood: 0,
              state: 'idle'
            })
            break
          }
        }
      })
    }, 100) // Update every 100ms

    return () => clearInterval(interval)
  }, [workers, trees, updateWorker, moveWorker, chopTree, setTreeBeingChopped, getAvailableTree, addResource])

  return null
}
