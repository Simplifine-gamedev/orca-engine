'use client'

import { useEffect } from 'react'
import ResourceBar from '../src/ui/ResourceBar'
import TreeSystem from '../src/resources/TreeSystem'
import WorkerRenderer from '../src/systems/WorkerRenderer'
import BuildingRenderer from '../src/systems/BuildingRenderer'
import WorkerAI from '../src/systems/WorkerAI'
import TreeRegrowth from '../src/systems/TreeRegrowth'
import { useTreeStore } from '../src/store/treeStore'
import { useWorkerStore } from '../src/store/workerStore'

export default function GamePage() {
  const initializeTrees = useTreeStore(state => state.initializeTrees)
  const initializeWorkers = useWorkerStore(state => state.initializeWorkers)

  useEffect(() => {
    // Initialize game on mount
    initializeTrees()
    initializeWorkers()
  }, [initializeTrees, initializeWorkers])

  return (
    <div className="w-screen h-screen overflow-hidden">
      <ResourceBar />
      
      <div className="relative w-full h-full bg-gradient-to-br from-green-900 via-green-800 to-green-900">
        {/* Game Map */}
        <div className="absolute inset-0 mt-20">
          {/* Ground texture */}
          <div className="absolute inset-0 opacity-20">
            <div className="w-full h-full" style={{
              backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 50px, rgba(0,0,0,0.1) 50px, rgba(0,0,0,0.1) 51px), repeating-linear-gradient(90deg, transparent, transparent 50px, rgba(0,0,0,0.1) 50px, rgba(0,0,0,0.1) 51px)'
            }} />
          </div>

          {/* Game entities */}
          <BuildingRenderer />
          <TreeSystem />
          <WorkerRenderer />
        </div>

        {/* Game systems (invisible) */}
        <WorkerAI />
        <TreeRegrowth />

        {/* Instructions */}
        <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white p-4 rounded-lg max-w-md">
          <h3 className="font-bold text-lg mb-2">🪵 Wood Gathering System</h3>
          <ul className="text-sm space-y-1">
            <li>• Workers automatically chop nearby trees</li>
            <li>• Trees contain wood that can be gathered</li>
            <li>• Workers return wood to the Town Hall</li>
            <li>• Trees regrow over time (10 wood every 5 seconds)</li>
            <li>• Hover over entities to see their status</li>
          </ul>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-black bg-opacity-70 text-white p-4 rounded-lg">
          <h3 className="font-bold text-sm mb-2">Legend</h3>
          <div className="text-xs space-y-1">
            <div>🏛️ Town Hall - Deposit point</div>
            <div>🌳 Tree - Wood source</div>
            <div>👷 Worker - Gathers wood</div>
            <div>⚒️ Chopping wood</div>
            <div>➡️ Moving to tree</div>
            <div>🏠 Returning home</div>
          </div>
        </div>
      </div>
    </div>
  )
}
