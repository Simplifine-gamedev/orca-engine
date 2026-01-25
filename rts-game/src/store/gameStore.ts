import { create } from 'zustand'
import { Resources, GameState } from '../types'

interface GameStore extends GameState {
  addResource: (type: keyof Resources, amount: number) => void
  removeResource: (type: keyof Resources, amount: number) => boolean
  canAfford: (cost: Partial<Resources>) => boolean
}

export const useGameStore = create<GameStore>((set, get) => ({
  resources: {
    wood: 100,
    gold: 100,
    food: 100
  },
  population: 5,
  maxPopulation: 10,

  addResource: (type, amount) => {
    set(state => ({
      resources: {
        ...state.resources,
        [type]: state.resources[type] + amount
      }
    }))
  },

  removeResource: (type, amount) => {
    const state = get()
    if (state.resources[type] >= amount) {
      set(state => ({
        resources: {
          ...state.resources,
          [type]: state.resources[type] - amount
        }
      }))
      return true
    }
    return false
  },

  canAfford: (cost) => {
    const state = get()
    return Object.entries(cost).every(([resource, amount]) => {
      const resourceType = resource as keyof Resources
      return state.resources[resourceType] >= (amount || 0)
    })
  }
}))
