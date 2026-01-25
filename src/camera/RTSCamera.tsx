import { useRef, useEffect } from 'react';
import * as THREE from 'three';

interface RTSCameraProps {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  minDistance?: number;
  maxDistance?: number;
  initialDistance?: number;
}

/**
 * RTS Camera Component
 * 
 * Provides an RTS-style camera with zoom, pan, and rotation controls.
 * 
 * @param minDistance - Minimum zoom distance (default: 1) - Reduced to allow closer zoom
 * @param maxDistance - Maximum zoom distance (default: 50)
 * @param initialDistance - Initial camera distance (default: 20)
 */
export const RTSCamera: React.FC<RTSCameraProps> = ({
  canvasRef,
  minDistance = 1,
  maxDistance = 50,
  initialDistance = 20,
}) => {
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsStateRef = useRef({
    distance: initialDistance,
    rotation: 0,
    pitch: Math.PI / 4, // 45 degrees
    target: new THREE.Vector3(0, 0, 0),
    isDragging: false,
    lastMousePos: { x: 0, y: 0 },
  });

  useEffect(() => {
    if (!canvasRef.current) return;

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    cameraRef.current = camera;

    // Update camera position based on controls
    const updateCameraPosition = () => {
      const state = controlsStateRef.current;
      const x = state.target.x + state.distance * Math.sin(state.rotation) * Math.cos(state.pitch);
      const y = state.target.y + state.distance * Math.sin(state.pitch);
      const z = state.target.z + state.distance * Math.cos(state.rotation) * Math.cos(state.pitch);
      
      camera.position.set(x, y, z);
      camera.lookAt(state.target);
    };

    updateCameraPosition();

    // Handle mouse wheel for zoom
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const state = controlsStateRef.current;
      
      // Zoom speed
      const zoomSpeed = 0.1;
      const delta = event.deltaY * zoomSpeed;
      
      // Update distance with constraints
      state.distance = Math.max(
        minDistance,
        Math.min(maxDistance, state.distance + delta)
      );
      
      updateCameraPosition();
    };

    // Handle mouse down for pan/rotate
    const handleMouseDown = (event: MouseEvent) => {
      const state = controlsStateRef.current;
      state.isDragging = true;
      state.lastMousePos = { x: event.clientX, y: event.clientY };
    };

    // Handle mouse move for pan/rotate
    const handleMouseMove = (event: MouseEvent) => {
      const state = controlsStateRef.current;
      if (!state.isDragging) return;

      const deltaX = event.clientX - state.lastMousePos.x;
      const deltaY = event.clientY - state.lastMousePos.y;

      // Right mouse button or ctrl+drag for rotation
      if (event.button === 2 || event.ctrlKey) {
        state.rotation += deltaX * 0.01;
        state.pitch = Math.max(
          0.1,
          Math.min(Math.PI / 2 - 0.1, state.pitch - deltaY * 0.01)
        );
      } else {
        // Pan
        const panSpeed = 0.02 * state.distance;
        const right = new THREE.Vector3(
          Math.cos(state.rotation),
          0,
          -Math.sin(state.rotation)
        );
        const forward = new THREE.Vector3(
          Math.sin(state.rotation),
          0,
          Math.cos(state.rotation)
        );

        state.target.add(right.multiplyScalar(-deltaX * panSpeed));
        state.target.add(forward.multiplyScalar(deltaY * panSpeed));
      }

      state.lastMousePos = { x: event.clientX, y: event.clientY };
      updateCameraPosition();
    };

    // Handle mouse up
    const handleMouseUp = () => {
      const state = controlsStateRef.current;
      state.isDragging = false;
    };

    // Handle window resize
    const handleResize = () => {
      if (!camera) return;
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    };

    // Add event listeners
    const canvas = canvasRef.current;
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('resize', handleResize);
    };
  }, [canvasRef, minDistance, maxDistance, initialDistance]);

  return null;
};

export default RTSCamera;
