'use client'

import { useState } from 'react'
import RTSCamera from '@/src/camera/RTSCamera'

export default function CameraDemoPage() {
  const [speed, setSpeed] = useState(5)
  const [multiplier, setMultiplier] = useState(2.5)
  const [cameraPos, setCameraPos] = useState({ x: 0, y: 0 })

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <h1 className="text-2xl font-bold mb-2">RTS Camera Demo</h1>
        <p className="text-gray-400 text-sm">
          Use WASD or Arrow keys to move. Hold SHIFT for faster movement.
        </p>
      </div>

      {/* Controls Panel */}
      <div className="bg-gray-800 border-b border-gray-700 p-4 flex gap-6 items-center">
        <div className="flex items-center gap-2">
          <label className="text-sm">Base Speed:</label>
          <input
            type="range"
            min="1"
            max="20"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-32"
          />
          <span className="text-sm font-mono w-8">{speed}</span>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm">SHIFT Multiplier:</label>
          <input
            type="range"
            min="1.5"
            max="5"
            step="0.1"
            value={multiplier}
            onChange={(e) => setMultiplier(Number(e.target.value))}
            className="w-32"
          />
          <span className="text-sm font-mono w-8">{multiplier.toFixed(1)}x</span>
        </div>

        <div className="ml-auto">
          <button
            onClick={() => setCameraPos({ x: 0, y: 0 })}
            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm"
          >
            Reset Camera
          </button>
        </div>
      </div>

      {/* Camera Stats */}
      <div className="bg-gray-800 border-b border-gray-700 p-3 flex gap-6 text-sm">
        <div>
          <span className="text-gray-400">Camera Position:</span>{' '}
          <span className="font-mono">
            ({cameraPos.x.toFixed(0)}, {cameraPos.y.toFixed(0)})
          </span>
        </div>
        <div>
          <span className="text-gray-400">Normal Speed:</span>{' '}
          <span className="font-mono">{speed} px/frame</span>
        </div>
        <div>
          <span className="text-gray-400">SHIFT Speed:</span>{' '}
          <span className="font-mono">{(speed * multiplier).toFixed(1)} px/frame</span>
        </div>
      </div>

      {/* Game Area with Camera */}
      <div className="flex-1 relative overflow-hidden">
        <RTSCamera
          basePanSpeed={speed}
          shiftSpeedMultiplier={multiplier}
          onPositionChange={(x, y) => setCameraPos({ x, y })}
        >
          {/* Demo game world with grid */}
          <div className="relative" style={{ width: '4000px', height: '4000px' }}>
            {/* Grid background */}
            <svg
              width="4000"
              height="4000"
              className="absolute inset-0"
              style={{ background: '#1a1a2e' }}
            >
              {/* Grid lines */}
              {Array.from({ length: 41 }, (_, i) => i * 100).map((pos) => (
                <g key={`grid-${pos}`}>
                  {/* Vertical lines */}
                  <line
                    x1={pos}
                    y1={0}
                    x2={pos}
                    y2={4000}
                    stroke="#2a2a4e"
                    strokeWidth={pos % 500 === 0 ? 2 : 1}
                  />
                  {/* Horizontal lines */}
                  <line
                    x1={0}
                    y1={pos}
                    x2={4000}
                    y2={pos}
                    stroke="#2a2a4e"
                    strokeWidth={pos % 500 === 0 ? 2 : 1}
                  />
                </g>
              ))}

              {/* Origin marker */}
              <circle cx={2000} cy={2000} r={10} fill="#ff6b6b" />
              <text x={2015} y={2005} fill="#ff6b6b" fontSize={14}>
                Origin (0, 0)
              </text>

              {/* Cardinal direction markers */}
              <text x={2000} y={1900} fill="#4ecdc4" fontSize={20} textAnchor="middle">
                N (W/↑)
              </text>
              <text x={2000} y={2120} fill="#4ecdc4" fontSize={20} textAnchor="middle">
                S (S/↓)
              </text>
              <text x={1850} y={2010} fill="#4ecdc4" fontSize={20} textAnchor="end">
                W (A/←)
              </text>
              <text x={2150} y={2010} fill="#4ecdc4" fontSize={20}>
                E (D/→)
              </text>

              {/* Demo objects scattered around */}
              {[
                { x: 1700, y: 1700, color: '#95e1d3', label: 'Base 1' },
                { x: 2300, y: 1700, color: '#f38181', label: 'Base 2' },
                { x: 1700, y: 2300, color: '#aa96da', label: 'Resource' },
                { x: 2300, y: 2300, color: '#fcbad3', label: 'Outpost' },
                { x: 2000, y: 1500, color: '#ffffd2', label: 'City' },
                { x: 1500, y: 2000, color: '#a8d8ea', label: 'Port' },
                { x: 2500, y: 2000, color: '#ffcfdf', label: 'Mine' }
              ].map((obj, i) => (
                <g key={`obj-${i}`}>
                  <rect
                    x={obj.x - 40}
                    y={obj.y - 40}
                    width={80}
                    height={80}
                    fill={obj.color}
                    stroke="#fff"
                    strokeWidth={2}
                  />
                  <text
                    x={obj.x}
                    y={obj.y + 5}
                    fill="#000"
                    fontSize={12}
                    textAnchor="middle"
                    fontWeight="bold"
                  >
                    {obj.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </RTSCamera>
      </div>

      {/* Instructions Footer */}
      <div className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="grid grid-cols-3 gap-6 text-sm">
          <div>
            <h3 className="font-semibold mb-2 text-blue-400">Keyboard Controls</h3>
            <ul className="space-y-1 text-gray-300">
              <li>• W / ↑ - Move Up</li>
              <li>• S / ↓ - Move Down</li>
              <li>• A / ← - Move Left</li>
              <li>• D / → - Move Right</li>
              <li>• SHIFT + Movement - Move Faster</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-2 text-green-400">Features</h3>
            <ul className="space-y-1 text-gray-300">
              <li>• Smooth camera panning</li>
              <li>• Normalized diagonal movement</li>
              <li>• Configurable speed settings</li>
              <li>• SHIFT speed boost (2-3x)</li>
              <li>• Position tracking</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-2 text-purple-400">Implementation</h3>
            <ul className="space-y-1 text-gray-300">
              <li>• React component</li>
              <li>• requestAnimationFrame</li>
              <li>• Efficient key tracking</li>
              <li>• TypeScript support</li>
              <li>• Customizable props</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
