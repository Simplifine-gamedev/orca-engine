'use client'

import { TOWN_HALL_POSITION } from '../store/workerStore'

export default function BuildingRenderer() {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Town Hall */}
      <div
        className="absolute pointer-events-auto"
        style={{
          left: TOWN_HALL_POSITION.x,
          top: TOWN_HALL_POSITION.y,
          transform: 'translate(-50%, -50%)'
        }}
        title="Town Hall - Resource deposit point"
      >
        <div className="relative">
          <div className="w-16 h-16 bg-stone-700 border-4 border-stone-900 flex items-center justify-center text-3xl rounded">
            🏛️
          </div>
          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-xs font-bold text-white bg-black bg-opacity-70 px-2 py-1 rounded whitespace-nowrap">
            Town Hall
          </div>
        </div>
      </div>
    </div>
  )
}
