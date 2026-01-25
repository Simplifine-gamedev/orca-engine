import { create } from 'zustand'
import { Worker, Position } from '../types'

interface WorkerStore {
  workers: Worker[]
  initializeWorkers: () => void
  updateWorker: (id: string, updates: Partial<Worker>) => void
  moveWorker: (id: string, target: Position) => void
  getWorker: (id: string) => Worker | undefined
}

const TOWN_HALL_POSITION: Position = { x: 400, y: 500 }

export const useWorkerStore = create<WorkerStore>((set, get) => ({
  workers: [],

  initializeWorkers: () => {
    const workers: Worker[] = []
    for (let i = 0; i < 3; i++) {
      workers.push({
        id: `worker-${i}`,
        position: { 
          x: TOWN_HALL_POSITION.x + (i - 1) * 30, 
          y: TOWN_HALL_POSITION.y 
        },
        targetTreeId: null,
        carryingWood: 0,
        maxCarryCapacity: 10,
        chopSpeed: 5,
        moveSpeed: 2,
        state: 'idle'
      })
    }
    set({ workers })
  },

  updateWorker: (id, updates) => {
    set(state => ({
      workers: state.workers.map(w =>
        w.id === id ? { ...w, ...updates } : w
      )
    }))
  },

  moveWorker: (id, target) => {
    set(state => ({
      workers: state.workers.map(w =>
        w.id === id ? { ...w, position: target } : w
      )
    }))
  },

  getWorker: (id) => {
    return get().workers.find(w => w.id === id)
  }
}))

export { TOWN_HALL_POSITION }
