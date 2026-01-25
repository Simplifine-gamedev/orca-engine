'use client'

import { useEffect, useRef, useState, useCallback } from 'react'

interface RTSCameraProps {
  /** Enable or disable edge panning */
  enabled?: boolean
  /** Distance from edge in pixels to trigger panning */
  edgeThreshold?: number
  /** Maximum pan speed in pixels per frame */
  maxPanSpeed?: number
  /** Minimum pan speed in pixels per frame */
  minPanSpeed?: number
  /** Callback when camera position changes */
  onCameraMove?: (deltaX: number, deltaY: number) => void
  /** Container class name */
  className?: string
  /** Children to render inside the camera viewport */
  children?: React.ReactNode
}

/**
 * RTSCamera component that implements edge-of-screen camera panning
 * for RTS-style games. When the mouse moves near the edges of the viewport,
 * the camera pans in that direction.
 */
export default function RTSCamera({
  enabled = true,
  edgeThreshold = 50,
  maxPanSpeed = 10,
  minPanSpeed = 2,
  onCameraMove,
  className = '',
  children
}: RTSCameraProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 })
  const animationFrameRef = useRef<number>()

  // Update viewport size on mount and resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setViewportSize({ width: rect.width, height: rect.height })
      }
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // Calculate pan speed based on distance from edge
  const calculatePanSpeed = useCallback(
    (distanceFromEdge: number): number => {
      if (distanceFromEdge > edgeThreshold) return 0

      // Linear interpolation between minPanSpeed and maxPanSpeed
      // Closer to edge = faster
      const ratio = 1 - distanceFromEdge / edgeThreshold
      return minPanSpeed + (maxPanSpeed - minPanSpeed) * ratio
    },
    [edgeThreshold, maxPanSpeed, minPanSpeed]
  )

  // Calculate camera movement based on mouse position
  const calculateCameraMovement = useCallback(() => {
    if (!enabled || !containerRef.current) return { deltaX: 0, deltaY: 0 }

    const rect = containerRef.current.getBoundingClientRect()
    const relativeX = mousePos.x - rect.left
    const relativeY = mousePos.y - rect.top

    // Check if mouse is outside the viewport
    if (
      relativeX < 0 ||
      relativeY < 0 ||
      relativeX > rect.width ||
      relativeY > rect.height
    ) {
      return { deltaX: 0, deltaY: 0 }
    }

    let deltaX = 0
    let deltaY = 0

    // Left edge
    if (relativeX < edgeThreshold) {
      const speed = calculatePanSpeed(relativeX)
      deltaX = -speed
    }
    // Right edge
    else if (relativeX > rect.width - edgeThreshold) {
      const speed = calculatePanSpeed(rect.width - relativeX)
      deltaX = speed
    }

    // Top edge
    if (relativeY < edgeThreshold) {
      const speed = calculatePanSpeed(relativeY)
      deltaY = -speed
    }
    // Bottom edge
    else if (relativeY > rect.height - edgeThreshold) {
      const speed = calculatePanSpeed(rect.height - relativeY)
      deltaY = speed
    }

    return { deltaX, deltaY }
  }, [enabled, mousePos, edgeThreshold, calculatePanSpeed])

  // Animation loop for smooth panning
  useEffect(() => {
    const animate = () => {
      const { deltaX, deltaY } = calculateCameraMovement()

      if ((deltaX !== 0 || deltaY !== 0) && onCameraMove) {
        onCameraMove(deltaX, deltaY)
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    if (enabled) {
      animationFrameRef.current = requestAnimationFrame(animate)
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [enabled, calculateCameraMovement, onCameraMove])

  // Track mouse movement
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY })
  }, [])

  // Track mouse position when entering the container
  const handleMouseEnter = useCallback((e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY })
  }, [])

  // Reset when mouse leaves
  const handleMouseLeave = useCallback(() => {
    setMousePos({ x: -1000, y: -1000 }) // Set far outside
  }, [])

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden ${className}`}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{ cursor: enabled ? 'default' : 'auto' }}
    >
      {children}

      {/* Visual indicators for edge zones (debug mode) */}
      {process.env.NODE_ENV === 'development' && enabled && (
        <>
          {/* Left edge */}
          <div
            className="absolute top-0 left-0 bottom-0 bg-blue-500 opacity-10 pointer-events-none"
            style={{ width: `${edgeThreshold}px` }}
          />
          {/* Right edge */}
          <div
            className="absolute top-0 right-0 bottom-0 bg-blue-500 opacity-10 pointer-events-none"
            style={{ width: `${edgeThreshold}px` }}
          />
          {/* Top edge */}
          <div
            className="absolute top-0 left-0 right-0 bg-blue-500 opacity-10 pointer-events-none"
            style={{ height: `${edgeThreshold}px` }}
          />
          {/* Bottom edge */}
          <div
            className="absolute bottom-0 left-0 right-0 bg-blue-500 opacity-10 pointer-events-none"
            style={{ height: `${edgeThreshold}px` }}
          />
        </>
      )}
    </div>
  )
}
