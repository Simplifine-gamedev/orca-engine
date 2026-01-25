import { useEffect, useRef, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { useGameStore } from './store/gameStore';
import { Workers } from './components/Worker';
import { Resources } from './components/Resource';
import { Buildings } from './buildings/Building';
import type * as THREE from 'three';

function Ground() {
  const { camera, raycaster, gl } = useThree();
  const groundRef = useRef<THREE.Mesh>(null);
  const updateBuildingGhostPosition = useGameStore((state) => state.updateBuildingGhostPosition);
  const buildingPlacement = useGameStore((state) => state.buildingPlacement);
  const confirmBuildingPlacement = useGameStore((state) => state.confirmBuildingPlacement);
  const cancelBuildingPlacement = useGameStore((state) => state.cancelBuildingPlacement);
  
  const handlePointerMove = (event: THREE.Event) => {
    if (!buildingPlacement.isActive) return;
    
    if (event.intersections && event.intersections.length > 0) {
      const point = event.intersections[0].point;
      // Snap to grid
      const snappedX = Math.round(point.x);
      const snappedZ = Math.round(point.z);
      updateBuildingGhostPosition({ x: snappedX, y: 0, z: snappedZ });
    }
  };
  
  const handleClick = (event: THREE.Event) => {
    if (buildingPlacement.isActive) {
      // Left click to place, right click to cancel
      if (event.nativeEvent.button === 0) {
        confirmBuildingPlacement();
      } else if (event.nativeEvent.button === 2) {
        cancelBuildingPlacement();
      }
    }
  };
  
  return (
    <>
      <mesh
        ref={groundRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        onPointerMove={handlePointerMove}
        onClick={handleClick}
        onContextMenu={(e) => e.stopPropagation()}
      >
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#4a6741" />
      </mesh>
      <Grid
        args={[100, 100]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#6e956c"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#8ab88a"
        fadeDistance={50}
        fadeStrength={1}
        followCamera={false}
      />
    </>
  );
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <Ground />
      <Workers />
      <Resources />
      <Buildings />
      <OrbitControls makeDefault />
    </>
  );
}

function UI() {
  const buildingPlacement = useGameStore((state) => state.buildingPlacement);
  const startBuildingPlacement = useGameStore((state) => state.startBuildingPlacement);
  const workers = useGameStore((state) => state.workers);
  const setWorkerState = useGameStore((state) => state.setWorkerState);
  const selectedWorker = useGameStore((state) => state.selectedWorker);
  const selectWorker = useGameStore((state) => state.selectWorker);
  
  const selectedWorkerData = workers.find((w) => w.id === selectedWorker);
  
  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      pointerEvents: 'none',
      fontFamily: 'Arial, sans-serif',
    }}>
      {/* Top bar */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        right: '10px',
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '15px',
        borderRadius: '8px',
        pointerEvents: 'auto',
      }}>
        <h2 style={{ margin: '0 0 10px 0' }}>Orca RTS - Building Preview Bug Demo</h2>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={() => startBuildingPlacement('barracks')}
            style={{
              padding: '8px 16px',
              background: buildingPlacement.type === 'barracks' ? '#4CAF50' : '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Build Barracks
          </button>
          <button
            onClick={() => startBuildingPlacement('farm')}
            style={{
              padding: '8px 16px',
              background: buildingPlacement.type === 'farm' ? '#4CAF50' : '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Build Farm
          </button>
          <button
            onClick={() => startBuildingPlacement('townhall')}
            style={{
              padding: '8px 16px',
              background: buildingPlacement.type === 'townhall' ? '#4CAF50' : '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Build Town Hall
          </button>
        </div>
        {buildingPlacement.isActive && (
          <p style={{ margin: '10px 0 0 0', color: '#4CAF50' }}>
            Click to place building | Right-click to cancel
          </p>
        )}
      </div>
      
      {/* Worker controls */}
      {selectedWorkerData && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '15px',
          borderRadius: '8px',
          pointerEvents: 'auto',
          minWidth: '200px',
        }}>
          <h3 style={{ margin: '0 0 10px 0' }}>Worker Selected</h3>
          <p style={{ margin: '5px 0' }}>State: {selectedWorkerData.state}</p>
          <div style={{ display: 'flex', gap: '5px', marginTop: '10px', flexDirection: 'column' }}>
            <button
              onClick={() => setWorkerState(selectedWorkerData.id, 'mining')}
              style={{
                padding: '6px 12px',
                background: selectedWorkerData.state === 'mining' ? '#4CAF50' : '#666',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Start Mining
            </button>
            <button
              onClick={() => setWorkerState(selectedWorkerData.id, 'idle')}
              style={{
                padding: '6px 12px',
                background: selectedWorkerData.state === 'idle' ? '#4CAF50' : '#666',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Stop Working
            </button>
            <button
              onClick={() => selectWorker(null)}
              style={{
                padding: '6px 12px',
                background: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                marginTop: '5px',
              }}
            >
              Deselect
            </button>
          </div>
        </div>
      )}
      
      {/* Instructions */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '15px',
        borderRadius: '8px',
        pointerEvents: 'auto',
        maxWidth: '300px',
      }}>
        <h3 style={{ margin: '0 0 10px 0' }}>Bug Reproduction Steps:</h3>
        <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
          <li>Select a worker (click on it)</li>
          <li>Set worker to mining state</li>
          <li>Try to place a building</li>
          <li>Building preview should appear (FIXED!)</li>
        </ol>
        <p style={{ margin: '10px 0 0 0', fontSize: '12px', color: '#4CAF50' }}>
          ✓ Bug Fix: Building ghost now shows regardless of worker state
        </p>
      </div>
    </div>
  );
}

function App() {
  const addWorker = useGameStore((state) => state.addWorker);
  const addResource = useGameStore((state) => state.addResource);
  const selectWorker = useGameStore((state) => state.selectWorker);
  const workers = useGameStore((state) => state.workers);
  
  // Initialize game world
  useEffect(() => {
    // Add some workers
    addWorker({
      id: 'worker-1',
      position: { x: -2, y: 0, z: 0 },
      state: 'idle',
      carrying: 0,
    });
    
    addWorker({
      id: 'worker-2',
      position: { x: 2, y: 0, z: 0 },
      state: 'idle',
      carrying: 0,
    });
    
    // Add gold mine
    addResource({
      id: 'gold-1',
      type: 'gold',
      position: { x: -5, y: 0, z: -5 },
      amount: 1000,
    });
    
    addResource({
      id: 'gold-2',
      type: 'gold',
      position: { x: 5, y: 0, z: 5 },
      amount: 1000,
    });
    
    // Select first worker by default
    setTimeout(() => selectWorker('worker-1'), 100);
  }, []);
  
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#1a1a1a' }}>
      <Canvas
        camera={{ position: [15, 15, 15], fov: 50 }}
        onPointerMissed={() => selectWorker(null)}
      >
        <Scene />
      </Canvas>
      <UI />
    </div>
  );
}

export default App;
