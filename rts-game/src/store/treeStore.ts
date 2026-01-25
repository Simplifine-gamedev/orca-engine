import { create } from 'zustand'
import { Tree } from '../types'

interface TreeStore {
  trees: Tree[]
  initializeTrees: () => void
  chopTree: (treeId: string, amount: number) => number
  setTreeBeingChopped: (treeId: string, isBeingChopped: boolean) => void
  updateTrees: () => void
  getAvailableTree: (workerX: number, workerY: number) => Tree | null
}

const TREE_REGROWTH_INTERVAL = 5000 // 5 seconds
const REGROWTH_AMOUNT = 10

export const useTreeStore = create<TreeStore>((set, get) => ({
  trees: [],

  initializeTrees: () => {
    // Create initial trees scattered around the map
    const trees: Tree[] = []
    const positions = [
      { x: 150, y: 200 },
      { x: 300, y: 150 },
      { x: 450, y: 250 },
      { x: 200, y: 400 },
      { x: 500, y: 350 },
      { x: 350, y: 450 },
      { x: 100, y: 500 },
      { x: 550, y: 150 },
      { x: 250, y: 300 },
      { x: 400, y: 100 }
    ]

    positions.forEach((pos, i) => {
      trees.push({
        id: `tree-${i}`,
        position: pos,
        wood: 100,
        maxWood: 100,
        regrowthRate: REGROWTH_AMOUNT,
        isBeingChopped: false,
        lastChopTime: Date.now()
      })
    })

    set({ trees })
  },

  chopTree: (treeId, amount) => {
    const state = get()
    const tree = state.trees.find(t => t.id === treeId)
    if (!tree || tree.wood <= 0) return 0

    const actualAmount = Math.min(amount, tree.wood)
    
    set({
      trees: state.trees.map(t =>
        t.id === treeId
          ? { ...t, wood: t.wood - actualAmount, lastChopTime: Date.now() }
          : t
      )
    })

    return actualAmount
  },

  setTreeBeingChopped: (treeId, isBeingChopped) => {
    set(state => ({
      trees: state.trees.map(t =>
        t.id === treeId ? { ...t, isBeingChopped } : t
      )
    }))
  },

  updateTrees: () => {
    const now = Date.now()
    set(state => ({
      trees: state.trees.map(tree => {
        if (tree.wood < tree.maxWood && now - tree.lastChopTime >= TREE_REGROWTH_INTERVAL) {
          return {
            ...tree,
            wood: Math.min(tree.maxWood, tree.wood + tree.regrowthRate),
            lastChopTime: now
          }
        }
        return tree
      })
    }))
  },

  getAvailableTree: (workerX, workerY) => {
    const state = get()
    const availableTrees = state.trees.filter(tree => tree.wood > 0 && !tree.isBeingChopped)
    
    if (availableTrees.length === 0) return null

    // Find closest tree
    return availableTrees.reduce((closest, tree) => {
      const distToTree = Math.hypot(tree.position.x - workerX, tree.position.y - workerY)
      const distToClosest = Math.hypot(closest.position.x - workerX, closest.position.y - workerY)
      return distToTree < distToClosest ? tree : closest
    })
  }
}))
