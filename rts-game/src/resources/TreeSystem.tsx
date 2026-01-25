'use client'

import { useTreeStore } from '../store/treeStore'

export default function TreeSystem() {
  const trees = useTreeStore(state => state.trees)

  return (
    <div className="absolute inset-0 pointer-events-none">
      {trees.map(tree => {
        const healthPercent = (tree.wood / tree.maxWood) * 100
        const isEmpty = tree.wood <= 0
        
        return (
          <div
            key={tree.id}
            className="absolute pointer-events-auto"
            style={{
              left: tree.position.x,
              top: tree.position.y,
              transform: 'translate(-50%, -50%)'
            }}
            title={`Wood: ${tree.wood}/${tree.maxWood}`}
          >
            {/* Tree visual */}
            <div className="relative">
              {/* Tree crown */}
              <div 
                className={`w-12 h-12 rounded-full transition-all ${
                  isEmpty 
                    ? 'bg-gray-400' 
                    : tree.isBeingChopped 
                      ? 'bg-yellow-600 animate-pulse' 
                      : 'bg-green-600'
                }`}
                style={{
                  opacity: isEmpty ? 0.3 : healthPercent / 100
                }}
              />
              {/* Tree trunk */}
              <div className="w-3 h-6 bg-amber-800 mx-auto" />
              
              {/* Wood amount indicator */}
              <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-xs font-bold text-white bg-black bg-opacity-70 px-2 py-1 rounded whitespace-nowrap">
                🪵 {tree.wood}
              </div>
              
              {/* Chopping indicator */}
              {tree.isBeingChopped && (
                <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-bold text-yellow-400 animate-bounce">
                  ⚒️
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
