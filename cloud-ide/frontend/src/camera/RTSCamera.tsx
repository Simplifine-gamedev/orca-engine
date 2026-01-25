'use client'

import { useEffect, useRef, useCallback, useState } from 'react'

export interface RTSCameraConfig {
  edgeThreshold?: number // Distance from edge in pixels to trigger panning
  baseSpeed?: number // Base panning speed
  maxSpeed?: number // Maximum panning speed at edge
  enabled?: boolean // Enable/disable edge panning
  smoothing?: number // Smoothing factor for camera movement (0-1)
}

export interface CameraPosition {
  x: number
  y: number
  z: number
}

export interface RTSCameraControls {
  position: CameraPosition
  setPosition: (pos: CameraPosition) => void
  velocity: { x: number; y: number }
  enabled: boolean
  setEnabled: (enabled: boolean) => void
}

const DEFAULT_CONFIG: Required<RTSCameraConfig> = {
  edgeThreshold: 50,
  baseSpeed: 5,
  maxSpeed: 20,
  enabled: true,
  smoothing: 0.85,
}

/**
 * RTS Camera Hook with Edge-of-Screen Panning
 * 
 * Implements standard RTS camera controls where moving the mouse
 * to the edges of the screen pans the camera in that direction.
 * 
 * @param containerRef - Reference to the container element for mouse tracking
 * @param config - Camera configuration options
 * @returns Camera controls and state
 */
export function useRTSCamera(
  containerRef: React.RefObject<HTMLElement>,
  config: RTSCameraConfig = {}
): RTSCameraControls {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config }
  const [position, setPosition] = useState<CameraPosition>({ x: 0, y: 0, z: 10 })
  const [velocity, setVelocity] = useState({ x: 0, y: 0 })
  const [enabled, setEnabled] = useState(mergedConfig.enabled)
  
  const velocityRef = useRef({ x: 0, y: 0 })
  const animationFrameRef = useRef<number>()

  /**
   * Calculate camera velocity based on mouse position relative to edges
   */
  const calculateEdgeVelocity = useCallback(
    (mouseX: number, mouseY: number, width: number, height: number) => {
      const { edgeThreshold, baseSpeed, maxSpeed } = mergedConfig
      let vx = 0
      let vy = 0

      // Left edge
      if (mouseX < edgeThreshold) {
        const ratio = 1 - mouseX / edgeThreshold
        vx = -baseSpeed - (maxSpeed - baseSpeed) * ratio
      }
      // Right edge
      else if (mouseX > width - edgeThreshold) {
        const ratio = (mouseX - (width - edgeThreshold)) / edgeThreshold
        vx = baseSpeed + (maxSpeed - baseSpeed) * ratio
      }

      // Top edge
      if (mouseY < edgeThreshold) {
        const ratio = 1 - mouseY / edgeThreshold
        vy = -baseSpeed - (maxSpeed - baseSpeed) * ratio
      }
      // Bottom edge
      else if (mouseY > height - edgeThreshold) {
        const ratio = (mouseY - (height - edgeThreshold)) / edgeThreshold
        vy = baseSpeed + (maxSpeed - baseSpeed) * ratio
      }

      return { x: vx, y: vy }
    },
    [mergedConfig]
  )

  /**
   * Handle mouse move events
   */
  useEffect(() => {
    const container = containerRef.current
    if (!container || !enabled) {
      velocityRef.current = { x: 0, y: 0 }
      setVelocity({ x: 0, y: 0 })
      return
    }

    const handleMouseMove = (event: MouseEvent) => {
      const rect = container.getBoundingClientRect()
      const mouseX = event.clientX - rect.left
      const mouseY = event.clientY - rect.top

      // Only calculate velocity if mouse is within the container
      if (
        mouseX >= 0 &&
        mouseX <= rect.width &&
        mouseY >= 0 &&
        mouseY <= rect.height
      ) {
        const newVelocity = calculateEdgeVelocity(
          mouseX,
          mouseY,
          rect.width,
          rect.height
        )
        velocityRef.current = newVelocity
        setVelocity(newVelocity)
      } else {
        velocityRef.current = { x: 0, y: 0 }
        setVelocity({ x: 0, y: 0 })
      }
    }

    const handleMouseLeave = () => {
      velocityRef.current = { x: 0, y: 0 }
      setVelocity({ x: 0, y: 0 })
    }

    container.addEventListener('mousemove', handleMouseMove)
    container.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      container.removeEventListener('mousemove', handleMouseMove)
      container.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [containerRef, enabled, calculateEdgeVelocity])

  /**
   * Animation loop to update camera position
   */
  useEffect(() => {
    if (!enabled) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      return
    }

    let lastTime = performance.now()

    const animate = (currentTime: number) => {
      const deltaTime = (currentTime - lastTime) / 1000 // Convert to seconds
      lastTime = currentTime

      const { x: vx, y: vy } = velocityRef.current

      // Apply velocity to position with delta time for frame-independent movement
      if (vx !== 0 || vy !== 0) {
        setPosition((prev) => ({
          x: prev.x + vx * deltaTime * 60, // Normalize to 60fps
          y: prev.y + vy * deltaTime * 60,
          z: prev.z,
        }))
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [enabled])

  return {
    position,
    setPosition,
    velocity,
    enabled,
    setEnabled,
  }
}

/**
 * RTS Camera Component
 * 
 * Wrapper component that provides edge-panning camera controls
 * for RTS-style games. Renders children and overlays camera controls.
 */
export interface RTSCameraProps {
  children: React.ReactNode
  config?: RTSCameraConfig
  onCameraMove?: (position: CameraPosition) => void
  className?: string
}

export default function RTSCamera({
  children,
  config = {},
  onCameraMove,
  className = '',
}: RTSCameraProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { position, velocity, enabled, setEnabled } = useRTSCamera(
    containerRef,
    config
  )

  // Notify parent component of camera position changes
  useEffect(() => {
    if (onCameraMove) {
      onCameraMove(position)
    }
  }, [position, onCameraMove])

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full ${className}`}
      style={{ cursor: enabled ? 'default' : 'not-allowed' }}
    >
      {children}
      
      {/* Debug overlay (optional, can be removed in production) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white text-xs p-2 rounded pointer-events-none">
          <div>Camera Position: ({position.x.toFixed(1)}, {position.y.toFixed(1)}, {position.z.toFixed(1)})</div>
          <div>Velocity: ({velocity.x.toFixed(1)}, {velocity.y.toFixed(1)})</div>
          <div>Edge Panning: {enabled ? 'ON' : 'OFF'}</div>
        </div>
      )}

      {/* Settings toggle button */}
      <button
        onClick={() => setEnabled(!enabled)}
        className="absolute top-2 right-2 bg-gray-800 hover:bg-gray-700 text-white text-xs px-3 py-1 rounded"
        title={`Edge panning: ${enabled ? 'enabled' : 'disabled'}`}
      >
        📹 {enabled ? 'ON' : 'OFF'}
      </button>
    </div>
  )
}
