import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { LoadingOverlay } from './ui/LoadingOverlay';
import { RTSUnit, registerRTSUnitAssets } from './units/RTSUnit';
import { NeutralMob, registerNeutralMobAssets } from './units/NeutralMob';
import { assetPreloader, LoadingProgress, PreloadedAssets } from './systems/AssetPreloader';

/**
 * Main App Component
 * 
 * This demonstrates the solution to ORC-100:
 * - Assets are preloaded during the loading screen
 * - Loading progress is shown for each asset type
 * - Game only starts when critical assets are loaded
 * - Models appear instantly without lag
 */
function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState<LoadingProgress>({
    loaded: 0,
    total: 0,
    currentAsset: '',
    percentage: 0,
    assetType: null,
  });
  const [loadingError, setLoadingError] = useState<string | null>(null);
  const [gameStarted, setGameStarted] = useState(false);

  useEffect(() => {
    initializeGame();
  }, []);

  /**
   * Initialize the game and preload all assets
   */
  const initializeGame = async () => {
    console.log('[App] Initializing game...');

    try {
      // Step 1: Register all assets that need to be preloaded
      registerGameAssets();

      // Step 2: Set up progress callback
      assetPreloader.setProgressCallback((progress) => {
        setLoadingProgress(progress);
      });

      // Step 3: Set up error callback
      assetPreloader.setErrorCallback((error, asset) => {
        console.error(`Failed to load ${asset.name}:`, error);
        if (asset.critical) {
          setLoadingError(`Critical asset failed to load: ${asset.name}`);
        }
      });

      // Step 4: Set up completion callback
      assetPreloader.setCompleteCallback((assets: PreloadedAssets) => {
        console.log('[App] All assets loaded:', assets);
        setIsLoading(false);
        setGameStarted(true);
      });

      // Step 5: Start preloading
      await assetPreloader.preloadAll();

    } catch (error) {
      console.error('[App] Failed to initialize game:', error);
      setLoadingError(error instanceof Error ? error.message : 'Unknown error');
      setIsLoading(false);
    }
  };

  /**
   * Register all game assets for preloading
   * 
   * This is where you list all the models, textures, and audio files
   * that need to be loaded before the game starts.
   */
  const registerGameAssets = () => {
    // Register RTS Unit models (workers, soldiers, etc.)
    // In a real game, these would be actual GLTF file URLs
    registerRTSUnitAssets([
      {
        name: 'worker',
        url: '/models/worker.glb',
        critical: true, // Workers are critical - mentioned in the bug report
      },
      {
        name: 'soldier',
        url: '/models/soldier.glb',
        critical: true,
      },
      {
        name: 'builder',
        url: '/models/builder.glb',
        critical: true,
      },
    ]);

    // Register Neutral Mob models
    registerNeutralMobAssets([
      {
        name: 'neutral_creature',
        url: '/models/neutral_creature.glb',
      },
      {
        name: 'neutral_resource',
        url: '/models/neutral_resource.glb',
      },
    ]);

    // Register additional assets
    assetPreloader.registerAssets([
      // Textures
      {
        name: 'terrain_texture',
        url: '/textures/terrain.png',
        type: 'texture',
        critical: false,
      },
      {
        name: 'ui_texture',
        url: '/textures/ui.png',
        type: 'texture',
        critical: false,
      },
      // Audio
      {
        name: 'background_music',
        url: '/audio/background.mp3',
        type: 'audio',
        critical: false,
      },
      {
        name: 'unit_select_sound',
        url: '/audio/select.mp3',
        type: 'audio',
        critical: false,
      },
    ]);
  };

  return (
    <>
      {/* Loading Overlay - shown during asset preloading */}
      <LoadingOverlay
        progress={loadingProgress}
        isLoading={isLoading}
        error={loadingError || undefined}
      />

      {/* Game Scene - only rendered after assets are loaded */}
      {gameStarted && !loadingError && (
        <div style={{ width: '100vw', height: '100vh' }}>
          <Canvas
            camera={{ position: [10, 10, 10], fov: 50 }}
            shadows
          >
            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} castShadow />

            {/* Scene Environment */}
            <Grid args={[20, 20]} />

            {/* RTS Units - these now load instantly! */}
            <RTSUnit
              position={[0, 0, 0]}
              modelName="worker"
              team="blue"
              unitType="worker"
            />
            <RTSUnit
              position={[3, 0, 0]}
              modelName="soldier"
              team="blue"
              unitType="soldier"
            />
            <RTSUnit
              position={[6, 0, 0]}
              modelName="builder"
              team="blue"
              unitType="builder"
            />

            <RTSUnit
              position={[0, 0, 6]}
              modelName="worker"
              team="red"
              unitType="worker"
            />
            <RTSUnit
              position={[3, 0, 6]}
              modelName="soldier"
              team="red"
              unitType="soldier"
            />

            {/* Neutral Mobs */}
            <NeutralMob
              position={[-5, 0, -5]}
              modelName="neutral_creature"
            />
            <NeutralMob
              position={[8, 0, -5]}
              modelName="neutral_resource"
            />

            {/* Camera Controls */}
            <OrbitControls />
          </Canvas>

          {/* UI Overlay */}
          <div
            style={{
              position: 'fixed',
              top: '20px',
              left: '20px',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              padding: '15px',
              borderRadius: '8px',
              color: 'white',
              fontFamily: 'Arial, sans-serif',
            }}
          >
            <h3 style={{ margin: '0 0 10px 0' }}>Orca RTS - Demo</h3>
            <div style={{ fontSize: '14px' }}>
              <div>✓ All models preloaded</div>
              <div>✓ No loading lag during gameplay</div>
              <div>✓ Units appear instantly</div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
