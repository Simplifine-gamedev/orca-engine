'use client'

import { useState } from 'react'
import RTSCamera from '../../components/RTSCamera'

export default function CameraDemo() {
  const [cameraPos, setCameraPos] = useState({ x: 0, y: 0 })
  const [enabled, setEnabled] = useState(true)
  const [edgeThreshold, setEdgeThreshold] = useState(50)
  const [maxSpeed, setMaxSpeed] = useState(10)

  const handleCameraMove = (deltaX: number, deltaY: number) => {
    setCameraPos((prev) => ({
      x: prev.x + deltaX,
      y: prev.y + deltaY
    }))
  }

  return (
    <div className="h-screen bg-gray-900 text-white flex">
      {/* Settings Panel */}
      <div className="w-80 bg-gray-800 p-6 border-r border-gray-700 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-6">RTS Camera Demo</h1>

        <div className="space-y-6">
          <div>
            <h2 className="text-lg font-semibold mb-3">Controls</h2>
            <p className="text-sm text-gray-300 mb-4">
              Move your mouse to the edges of the viewport to pan the camera.
            </p>
          </div>

          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
                className="w-4 h-4"
              />
              <span>Enable Edge Panning</span>
            </label>
          </div>

          <div>
            <label className="block mb-2">
              Edge Threshold: {edgeThreshold}px
            </label>
            <input
              type="range"
              min="20"
              max="150"
              value={edgeThreshold}
              onChange={(e) => setEdgeThreshold(Number(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-400 mt-1">
              Distance from edge to trigger panning
            </p>
          </div>

          <div>
            <label className="block mb-2">Max Speed: {maxSpeed}px/frame</label>
            <input
              type="range"
              min="1"
              max="30"
              value={maxSpeed}
              onChange={(e) => setMaxSpeed(Number(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-400 mt-1">
              Maximum camera pan speed
            </p>
          </div>

          <div>
            <h3 className="font-semibold mb-2">Camera Position</h3>
            <div className="bg-gray-900 p-3 rounded font-mono text-sm">
              <div>X: {cameraPos.x.toFixed(2)}</div>
              <div>Y: {cameraPos.y.toFixed(2)}</div>
            </div>
            <button
              onClick={() => setCameraPos({ x: 0, y: 0 })}
              className="mt-2 w-full bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm"
            >
              Reset Position
            </button>
          </div>

          <div className="border-t border-gray-700 pt-4">
            <h3 className="font-semibold mb-2">Features</h3>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>✓ Works on all 4 edges</li>
              <li>✓ Speed increases near edge</li>
              <li>✓ Smooth animation</li>
              <li>✓ Easy to enable/disable</li>
              <li>✓ Configurable thresholds</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Camera Viewport */}
      <div className="flex-1 relative">
        <RTSCamera
          enabled={enabled}
          edgeThreshold={edgeThreshold}
          maxPanSpeed={maxSpeed}
          minPanSpeed={2}
          onCameraMove={handleCameraMove}
          className="w-full h-full"
        >
          {/* Game World/Viewport Content */}
          <div className="w-full h-full bg-gradient-to-br from-gray-800 to-gray-900 relative overflow-hidden">
            {/* Grid background */}
            <div
              className="absolute inset-0"
              style={{
                backgroundImage: `
                  linear-gradient(to right, #374151 1px, transparent 1px),
                  linear-gradient(to bottom, #374151 1px, transparent 1px)
                `,
                backgroundSize: '50px 50px',
                transform: `translate(${cameraPos.x % 50}px, ${cameraPos.y % 50}px)`
              }}
            />

            {/* Sample game objects */}
            <div
              className="absolute w-16 h-16 bg-blue-500 rounded-lg shadow-lg flex items-center justify-center"
              style={{
                left: `${300 + cameraPos.x}px`,
                top: `${200 + cameraPos.y}px`
              }}
            >
              <span className="text-2xl">🏰</span>
            </div>

            <div
              className="absolute w-12 h-12 bg-green-500 rounded-full shadow-lg flex items-center justify-center"
              style={{
                left: `${500 + cameraPos.x}px`,
                top: `${300 + cameraPos.y}px`
              }}
            >
              <span className="text-xl">🌳</span>
            </div>

            <div
              className="absolute w-12 h-12 bg-red-500 rounded-lg shadow-lg flex items-center justify-center"
              style={{
                left: `${700 + cameraPos.x}px`,
                top: `${250 + cameraPos.y}px`
              }}
            >
              <span className="text-xl">⚔️</span>
            </div>

            <div
              className="absolute w-12 h-12 bg-yellow-500 rounded-full shadow-lg flex items-center justify-center"
              style={{
                left: `${400 + cameraPos.x}px`,
                top: `${450 + cameraPos.y}px`
              }}
            >
              <span className="text-xl">💎</span>
            </div>

            {/* Center indicator */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-4 h-4 border-2 border-white rounded-full opacity-50" />
            </div>

            {/* Instructions overlay */}
            {enabled && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-70 px-6 py-3 rounded-lg">
                <p className="text-sm text-center">
                  Move your mouse to the edges of the screen to pan the camera
                </p>
              </div>
            )}

            {!enabled && (
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-black bg-opacity-70 px-8 py-4 rounded-lg">
                <p className="text-center">
                  Edge panning is disabled
                  <br />
                  <span className="text-sm text-gray-400">
                    Enable it in the settings panel
                  </span>
                </p>
              </div>
            )}
          </div>
        </RTSCamera>
      </div>
    </div>
  )
}
