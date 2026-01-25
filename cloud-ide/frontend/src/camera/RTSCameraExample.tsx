'use client'

import { useRef } from 'react'
import RTSCamera, { useRTSCamera, CameraPosition } from './RTSCamera'

/**
 * Example 1: Using the RTSCamera component wrapper
 */
export function RTSCameraComponentExample() {
  const handleCameraMove = (position: CameraPosition) => {
    // This callback is fired whenever the camera position changes
    // You can use this to update your 3D engine camera, game state, etc.
    console.log('Camera position:', position)
  }

  return (
    <div className="w-full h-screen">
      <RTSCamera
        config={{
          edgeThreshold: 50,  // Trigger within 50px of edge
          baseSpeed: 5,       // Minimum speed
          maxSpeed: 20,       // Maximum speed at edge
          enabled: true,      // Start enabled
        }}
        onCameraMove={handleCameraMove}
        className="bg-gray-900"
      >
        {/* Your game viewport content goes here */}
        <div className="flex items-center justify-center h-full text-white">
          <div className="text-center">
            <h1 className="text-4xl mb-4">RTS Game Viewport</h1>
            <p className="text-gray-400">Move your mouse to the edges to pan the camera</p>
          </div>
        </div>
      </RTSCamera>
    </div>
  )
}

/**
 * Example 2: Using the useRTSCamera hook for more control
 */
export function RTSCameraHookExample() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { position, velocity, enabled, setEnabled } = useRTSCamera(
    containerRef,
    {
      edgeThreshold: 60,
      baseSpeed: 3,
      maxSpeed: 15,
      enabled: true,
    }
  )

  return (
    <div 
      ref={containerRef}
      className="w-full h-screen bg-gradient-to-br from-blue-900 to-purple-900 relative"
    >
      {/* Custom UI */}
      <div className="absolute top-4 left-4 bg-black bg-opacity-60 text-white p-4 rounded-lg">
        <h2 className="text-lg font-bold mb-2">Camera Info</h2>
        <div className="space-y-1 text-sm font-mono">
          <div>X: {position.x.toFixed(2)}</div>
          <div>Y: {position.y.toFixed(2)}</div>
          <div>Z: {position.z.toFixed(2)}</div>
          <div className="pt-2 border-t border-gray-600 mt-2">
            <div>Velocity X: {velocity.x.toFixed(2)}</div>
            <div>Velocity Y: {velocity.y.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Custom toggle */}
      <div className="absolute top-4 right-4">
        <button
          onClick={() => setEnabled(!enabled)}
          className={`px-4 py-2 rounded font-semibold transition-colors ${
            enabled 
              ? 'bg-green-600 hover:bg-green-700 text-white' 
              : 'bg-gray-600 hover:bg-gray-700 text-gray-300'
          }`}
        >
          Edge Panning: {enabled ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Game content - this would be your iframe, canvas, etc. */}
      <div className="flex items-center justify-center h-full text-white">
        <div className="text-center">
          <h1 className="text-4xl mb-4">Custom Hook Example</h1>
          <p className="text-gray-300">Move mouse to edges to pan</p>
          <div className="mt-8 p-6 bg-black bg-opacity-40 rounded-lg">
            <p className="text-sm">
              This example uses the useRTSCamera hook directly,
              <br />
              giving you full control over the UI and behavior.
            </p>
          </div>
        </div>
      </div>

      {/* Visual indicators for panning zones */}
      {enabled && (
        <>
          {/* Top edge indicator */}
          {velocity.y < 0 && (
            <div className="absolute top-0 left-0 right-0 h-1 bg-blue-500 opacity-70" />
          )}
          {/* Bottom edge indicator */}
          {velocity.y > 0 && (
            <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-500 opacity-70" />
          )}
          {/* Left edge indicator */}
          {velocity.x < 0 && (
            <div className="absolute top-0 left-0 bottom-0 w-1 bg-blue-500 opacity-70" />
          )}
          {/* Right edge indicator */}
          {velocity.x > 0 && (
            <div className="absolute top-0 right-0 bottom-0 w-1 bg-blue-500 opacity-70" />
          )}
        </>
      )}
    </div>
  )
}

/**
 * Example 3: Integration with Orca IDE viewport
 */
export function OrcaIDEIntegration() {
  const viewportRef = useRef<HTMLDivElement>(null)
  const { position, enabled, setEnabled } = useRTSCamera(viewportRef, {
    edgeThreshold: 40,
    baseSpeed: 8,
    maxSpeed: 25,
    enabled: true,
  })

  // In a real integration, you would send camera position updates
  // to the VNC/iframe or game engine
  const updateEngineCamera = (pos: CameraPosition) => {
    // Example: Send to VNC iframe
    // const iframe = document.querySelector('iframe')
    // iframe?.contentWindow?.postMessage({ type: 'camera-update', position: pos }, '*')
    
    // Or update via WebSocket
    // socket.emit('camera-update', pos)
  }

  return (
    <div 
      ref={viewportRef}
      className="relative w-full h-full bg-black"
    >
      {/* This div would contain your VNC iframe or game canvas */}
      <div className="w-full h-full flex items-center justify-center text-white">
        <div className="text-center">
          <div className="text-xl mb-2">3D Viewport</div>
          <div className="text-sm text-gray-400">
            Camera Position: ({position.x.toFixed(0)}, {position.y.toFixed(0)}, {position.z.toFixed(0)})
          </div>
          <div className="text-sm text-gray-500 mt-2">
            Move mouse to edges to pan camera
          </div>
        </div>
      </div>

      {/* Small settings button in viewport */}
      <button
        onClick={() => setEnabled(!enabled)}
        className="absolute bottom-4 right-4 bg-gray-800 bg-opacity-80 hover:bg-opacity-100 text-white text-xs px-3 py-1.5 rounded backdrop-blur-sm transition-all"
        title="Toggle edge panning"
      >
        🎥 Edge Pan: {enabled ? 'ON' : 'OFF'}
      </button>
    </div>
  )
}
