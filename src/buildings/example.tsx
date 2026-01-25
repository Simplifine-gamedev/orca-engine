/**
 * Example integration of the Wall Building System
 * This demonstrates how to use the wall system in your game
 */

import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import {
  WallPreview,
  useWallSystem,
  initializeWallSystem,
  cleanupWallSystem,
  WallLoadingIndicator,
} from './index';

/**
 * Main game component that initializes the wall system
 */
export function GameWithWallSystem() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    // Initialize wall system on component mount
    const init = async () => {
      try {
        await initializeWallSystem();
        setIsInitialized(true);
        console.log('Wall system initialized successfully');
      } catch (error) {
        console.error('Failed to initialize wall system:', error);
        setInitError(error instanceof Error ? error.message : 'Unknown error');
      }
    };

    init();

    // Cleanup on unmount
    return () => {
      cleanupWallSystem();
    };
  }, []);

  if (initError) {
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h2>Failed to initialize wall system</h2>
        <p>{initError}</p>
      </div>
    );
  }

  if (!isInitialized) {
    return <WallLoadingIndicator isLoading={true} message="Initializing game..." />;
  }

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <BuildingInterface />
    </div>
  );
}

/**
 * Building interface component with wall building controls
 */
function BuildingInterface() {
  const {
    isBuildMode,
    isPreloading,
    wallPreviews,
    enterBuildMode,
    exitBuildMode,
    addWallPreview,
    clearWallPreviews,
  } = useWallSystem();

  const [cursorPosition, setCursorPosition] = useState<[number, number, number]>([0, 0, 0]);

  const handleEnterBuildMode = async () => {
    console.log('Entering wall build mode...');
    await enterBuildMode();
    console.log('Wall build mode active');
  };

  const handleExitBuildMode = () => {
    console.log('Exiting wall build mode');
    exitBuildMode();
  };

  const handlePlaceWall = () => {
    if (!isBuildMode) return;

    addWallPreview({
      position: [...cursorPosition],
      type: 'wall_segment',
      isValid: true,
    });

    console.log(`Placed wall at position: ${cursorPosition.join(', ')}`);
  };

  const handleClearWalls = () => {
    clearWallPreviews();
    console.log('Cleared all wall previews');
  };

  return (
    <>
      {/* Loading indicator */}
      <WallLoadingIndicator
        isLoading={isPreloading}
        message="Preparing wall blueprints..."
      />

      {/* UI Controls */}
      <div
        style={{
          position: 'absolute',
          top: '20px',
          left: '20px',
          zIndex: 100,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '15px',
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'Arial, sans-serif',
        }}
      >
        <h3 style={{ margin: '0 0 10px 0' }}>Wall Building System</h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {!isBuildMode ? (
            <button
              onClick={handleEnterBuildMode}
              disabled={isPreloading}
              style={{
                padding: '10px 20px',
                backgroundColor: isPreloading ? '#666' : '#4488ff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isPreloading ? 'not-allowed' : 'pointer',
                fontSize: '14px',
              }}
            >
              {isPreloading ? 'Loading...' : 'Enter Build Mode'}
            </button>
          ) : (
            <>
              <button
                onClick={handleExitBuildMode}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#ff4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                Exit Build Mode
              </button>

              <button
                onClick={handlePlaceWall}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#44ff44',
                  color: 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                Place Wall
              </button>

              {wallPreviews.length > 0 && (
                <button
                  onClick={handleClearWalls}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#ff8844',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '14px',
                  }}
                >
                  Clear All ({wallPreviews.length})
                </button>
              )}
            </>
          )}
        </div>

        <div style={{ marginTop: '15px', fontSize: '12px', opacity: 0.8 }}>
          <div>Build Mode: {isBuildMode ? 'Active' : 'Inactive'}</div>
          <div>Wall Previews: {wallPreviews.length}</div>
          <div>Cursor: [{cursorPosition.map((v) => v.toFixed(1)).join(', ')}]</div>
        </div>
      </div>

      {/* 3D Scene */}
      <Canvas
        camera={{ position: [10, 10, 10], fov: 50 }}
        shadows
        onPointerMove={(e) => {
          // Update cursor position based on raycast to ground plane
          if (e.intersections.length > 0) {
            const intersection = e.intersections[0];
            const { x, y, z } = intersection.point;
            setCursorPosition([
              Math.round(x),
              Math.round(y),
              Math.round(z),
            ]);
          }
        }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />

        {/* Ground plane */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
          <planeGeometry args={[50, 50]} />
          <meshStandardMaterial color="#2a4a2a" />
        </mesh>

        {/* Cursor preview in build mode */}
        {isBuildMode && (
          <WallPreview
            position={cursorPosition}
            type="wall_segment"
            isValid={true}
          />
        )}

        {/* Render all placed wall previews */}
        {wallPreviews.map((preview, index) => (
          <WallPreview key={index} {...preview} />
        ))}

        {/* Camera controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={50}
        />
      </Canvas>
    </>
  );
}

export default GameWithWallSystem;
