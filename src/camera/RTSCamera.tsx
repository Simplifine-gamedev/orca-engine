import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

interface RTSCameraProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  minDistance?: number;
  maxDistance?: number;
  zoomSpeed?: number;
  moveSpeed?: number;
}

/**
 * RTS Camera Component
 * Provides camera controls for RTS-style games including:
 * - Zoom in/out with mouse wheel
 * - Pan with middle mouse or WASD keys
 * - Configurable min/max zoom distances
 */
export const RTSCamera: React.FC<RTSCameraProps> = ({
  canvasRef,
  minDistance = 2, // Reduced from default to allow closer zoom
  maxDistance = 50,
  zoomSpeed = 2,
  moveSpeed = 0.5,
}) => {
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const [distance, setDistance] = useState(20);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!canvasRef.current) return;

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    cameraRef.current = camera;

    // Set initial camera position (RTS angle)
    updateCameraPosition(camera, position.x, position.y, distance);

    // Handle mouse wheel for zooming
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = event.deltaY * 0.01;
      const newDistance = Math.max(
        minDistance,
        Math.min(maxDistance, distance + delta * zoomSpeed)
      );
      setDistance(newDistance);
      updateCameraPosition(camera, position.x, position.y, newDistance);
    };

    // Handle keyboard for panning
    const handleKeyDown = (event: KeyboardEvent) => {
      const step = moveSpeed;
      let newX = position.x;
      let newY = position.y;

      switch (event.key.toLowerCase()) {
        case 'w':
        case 'arrowup':
          newY += step;
          break;
        case 's':
        case 'arrowdown':
          newY -= step;
          break;
        case 'a':
        case 'arrowleft':
          newX -= step;
          break;
        case 'd':
        case 'arrowright':
          newX += step;
          break;
      }

      if (newX !== position.x || newY !== position.y) {
        setPosition({ x: newX, y: newY });
        updateCameraPosition(camera, newX, newY, distance);
      }
    };

    // Mouse drag for panning
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const handleMouseDown = (event: MouseEvent) => {
      if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
        // Middle mouse or Shift+Left mouse
        isDragging = true;
        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
        event.preventDefault();
      }
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (isDragging) {
        const deltaX = (event.clientX - lastMouseX) * 0.01;
        const deltaY = (event.clientY - lastMouseY) * 0.01;

        const newX = position.x - deltaX * (distance / 10);
        const newY = position.y + deltaY * (distance / 10);

        setPosition({ x: newX, y: newY });
        updateCameraPosition(camera, newX, newY, distance);

        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
      }
    };

    const handleMouseUp = (event: MouseEvent) => {
      if (event.button === 1 || event.button === 0) {
        isDragging = false;
      }
    };

    // Handle window resize
    const handleResize = () => {
      if (cameraRef.current) {
        cameraRef.current.aspect = window.innerWidth / window.innerHeight;
        cameraRef.current.updateProjectionMatrix();
      }
    };

    // Add event listeners
    const canvas = canvasRef.current;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', handleResize);
    };
  }, [canvasRef, distance, position, minDistance, maxDistance, zoomSpeed, moveSpeed]);

  return null; // This is a controller component, no visual output
};

/**
 * Update camera position based on RTS-style perspective
 * @param camera - Three.js camera instance
 * @param x - X position of the look-at point
 * @param y - Y position of the look-at point (forward/back)
 * @param distance - Distance from the look-at point
 */
function updateCameraPosition(
  camera: THREE.PerspectiveCamera,
  x: number,
  y: number,
  distance: number
) {
  // RTS camera angle (45 degrees looking down)
  const angle = Math.PI / 4; // 45 degrees
  const height = distance * Math.sin(angle);
  const horizontalDistance = distance * Math.cos(angle);

  // Position camera
  camera.position.set(x, height, y + horizontalDistance);

  // Look at the target point
  camera.lookAt(x, 0, y);

  camera.updateProjectionMatrix();
}

/**
 * Get the current camera instance (for external use)
 */
export function useRTSCamera() {
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  return {
    camera: cameraRef.current,
    getCamera: () => cameraRef.current,
  };
}

export default RTSCamera;
