'use client'

import { useWorkerStore } from '../store/workerStore'

export default function WorkerRenderer() {
  const workers = useWorkerStore(state => state.workers)

  return (
    <div className="absolute inset-0 pointer-events-none">
      {workers.map(worker => (
        <div
          key={worker.id}
          className="absolute pointer-events-auto"
          style={{
            left: worker.position.x,
            top: worker.position.y,
            transform: 'translate(-50%, -50%)'
          }}
          title={`${worker.state} | Wood: ${worker.carryingWood}/${worker.maxCarryCapacity}`}
        >
          {/* Worker visual */}
          <div className="relative">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-xs border-2 border-blue-700">
              👷
            </div>
            
            {/* State indicator */}
            <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs font-bold whitespace-nowrap">
              {worker.state === 'chopping' && '⚒️'}
              {worker.state === 'moving_to_tree' && '➡️'}
              {worker.state === 'returning' && '🏠'}
              {worker.carryingWood > 0 && (
                <span className="bg-amber-700 text-white px-1 rounded ml-1">
                  🪵{worker.carryingWood}
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
