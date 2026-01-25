'use client'

import { useGameStore } from '../store/gameStore'

export default function ResourceBar() {
  const resources = useGameStore(state => state.resources)
  const population = useGameStore(state => state.population)
  const maxPopulation = useGameStore(state => state.maxPopulation)

  return (
    <div className="fixed top-0 left-0 right-0 bg-gray-900 bg-opacity-90 text-white p-4 shadow-lg z-10">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-6">
          <h1 className="text-2xl font-bold text-blue-400">Orca RTS</h1>
          
          <div className="flex items-center gap-4">
            <ResourceItem icon="🪵" label="Wood" value={resources.wood} />
            <ResourceItem icon="🪙" label="Gold" value={resources.gold} />
            <ResourceItem icon="🌾" label="Food" value={resources.food} />
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="bg-gray-800 px-4 py-2 rounded-lg">
            <span className="text-gray-400">Population:</span>
            <span className="ml-2 font-bold text-green-400">
              {population}/{maxPopulation}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

function ResourceItem({ icon, label, value }: { icon: string; label: string; value: number }) {
  return (
    <div className="bg-gray-800 px-4 py-2 rounded-lg flex items-center gap-2 min-w-[120px]">
      <span className="text-xl">{icon}</span>
      <div className="flex flex-col">
        <span className="text-xs text-gray-400">{label}</span>
        <span className="text-lg font-bold text-yellow-400">{value}</span>
      </div>
    </div>
  )
}
