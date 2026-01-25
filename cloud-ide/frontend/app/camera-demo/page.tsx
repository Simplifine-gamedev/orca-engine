'use client'

import { useState } from 'react'
import {
  RTSCameraComponentExample,
  RTSCameraHookExample,
  OrcaIDEIntegration,
} from '../../src/camera/RTSCameraExample'

type DemoMode = 'component' | 'hook' | 'ide'

export default function CameraDemoPage() {
  const [mode, setMode] = useState<DemoMode>('component')

  return (
    <div className="h-screen flex flex-col">
      {/* Demo selector */}
      <div className="bg-gray-800 text-white p-4 border-b border-gray-700">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold mb-3">RTS Camera Edge Panning Demo</h1>
          <p className="text-sm text-gray-400 mb-4">
            Move your mouse to the edges of the viewport to pan the camera. 
            Speed increases as you get closer to the edge.
          </p>
          
          <div className="flex gap-2">
            <button
              onClick={() => setMode('component')}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                mode === 'component'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Component Example
            </button>
            <button
              onClick={() => setMode('hook')}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                mode === 'hook'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Hook Example
            </button>
            <button
              onClick={() => setMode('ide')}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                mode === 'ide'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              IDE Integration
            </button>
          </div>
        </div>
      </div>

      {/* Demo viewport */}
      <div className="flex-1 overflow-hidden">
        {mode === 'component' && <RTSCameraComponentExample />}
        {mode === 'hook' && <RTSCameraHookExample />}
        {mode === 'ide' && <OrcaIDEIntegration />}
      </div>
    </div>
  )
}
