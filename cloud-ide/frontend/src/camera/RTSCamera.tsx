'use client'

import { useEffect, useRef, useState } from 'react'

interface RTSCameraProps {
  /**
   * Base pan speed for camera movement (pixels per frame)
   * @default 5
   */
  basePanSpeed?: number
  
  /**
   * Speed multiplier when SHIFT key is held
   * @default 2.5
   */
  shiftSpeedMultiplier?: number
  
  /**
   * Optional callback when camera position changes
   */
  onPositionChange?: (x: number, y: number) => void
  
  /**
   * Children to render (typically the game viewport)
   */
  children?: React.ReactNode
}

interface CameraState {
  x: number
  y: number
  isShiftHeld: boolean
}

/**
 * RTS-style camera component with keyboard controls
 * 
 * Features:
 * - WASD and Arrow keys for movement
 * - SHIFT key for faster movement (2-3x speed)
 * - Smooth camera panning
 * 
 * Usage:
 * ```tsx
 * <RTSCamera basePanSpeed={5} shiftSpeedMultiplier={2.5}>
 *   <GameViewport />
 * </RTSCamera>
 * ```
 */
export default function RTSCamera({
  basePanSpeed = 5,
  shiftSpeedMultiplier = 2.5,
  onPositionChange,
  children
}: RTSCameraProps) {
  const [cameraState, setCameraState] = useState<CameraState>({
    x: 0,
    y: 0,
    isShiftHeld: false
  })
  
  const keysPressed = useRef<Set<string>>(new Set())
  const animationFrameId = useRef<number>()

  // Handle keyboard input
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      
      // Track SHIFT key state
      if (e.shiftKey) {
        setCameraState(prev => ({ ...prev, isShiftHeld: true }))
      }
      
      // Track movement keys (WASD and Arrow keys)
      if (
        ['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)
      ) {
        keysPressed.current.add(key)
        e.preventDefault() // Prevent default browser behavior
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase()
      
      // Track SHIFT key state
      if (key === 'shift') {
        setCameraState(prev => ({ ...prev, isShiftHeld: false }))
      }
      
      // Remove key from pressed set
      keysPressed.current.delete(key)
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
    }
  }, [])

  // Camera update loop
  useEffect(() => {
    const updateCamera = () => {
      const keys = keysPressed.current
      
      if (keys.size === 0) {
        animationFrameId.current = requestAnimationFrame(updateCamera)
        return
      }

      setCameraState(prev => {
        let deltaX = 0
        let deltaY = 0

        // Calculate movement direction
        // Vertical movement (W/S or Up/Down arrows)
        if (keys.has('w') || keys.has('arrowup')) {
          deltaY -= 1
        }
        if (keys.has('s') || keys.has('arrowdown')) {
          deltaY += 1
        }

        // Horizontal movement (A/D or Left/Right arrows)
        if (keys.has('a') || keys.has('arrowleft')) {
          deltaX -= 1
        }
        if (keys.has('d') || keys.has('arrowright')) {
          deltaX += 1
        }

        // Normalize diagonal movement to prevent faster diagonal speed
        if (deltaX !== 0 && deltaY !== 0) {
          const normalizer = Math.sqrt(2)
          deltaX /= normalizer
          deltaY /= normalizer
        }

        // Apply speed multiplier
        const currentSpeed = prev.isShiftHeld 
          ? basePanSpeed * shiftSpeedMultiplier 
          : basePanSpeed

        // Calculate new position
        const newX = prev.x + deltaX * currentSpeed
        const newY = prev.y + deltaY * currentSpeed

        // Notify position change
        if (onPositionChange && (newX !== prev.x || newY !== prev.y)) {
          onPositionChange(newX, newY)
        }

        return {
          ...prev,
          x: newX,
          y: newY
        }
      })

      animationFrameId.current = requestAnimationFrame(updateCamera)
    }

    animationFrameId.current = requestAnimationFrame(updateCamera)

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
  }, [basePanSpeed, shiftSpeedMultiplier, onPositionChange])

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden'
      }}
    >
      <div
        style={{
          transform: `translate(${-cameraState.x}px, ${-cameraState.y}px)`,
          transition: 'transform 0.016s linear' // ~60fps
        }}
      >
        {children}
      </div>
      
      {/* Debug info (can be removed in production) */}
      {process.env.NODE_ENV === 'development' && (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: 4,
            fontSize: 12,
            fontFamily: 'monospace',
            pointerEvents: 'none',
            zIndex: 1000
          }}
        >
          <div>Camera Position: ({cameraState.x.toFixed(1)}, {cameraState.y.toFixed(1)})</div>
          <div>
            Speed: {cameraState.isShiftHeld ? 'FAST' : 'NORMAL'} 
            {cameraState.isShiftHeld && ` (${shiftSpeedMultiplier}x)`}
          </div>
        </div>
      )}
    </div>
  )
}
