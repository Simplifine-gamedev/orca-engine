import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { assetPreloader, gltfCache } from '../systems/AssetPreloader';

interface NeutralMobProps {
  position: [number, number, number];
  modelName: string;
  scale?: number;
}

/**
 * NeutralMob - Neutral unit in the game
 * 
 * This component uses the GLTF cache to avoid re-downloading models.
 * Models are now preloaded during the loading screen, so they appear instantly.
 * 
 * Fix for: "took quite a bit of time for models (especially the workers) to load"
 */
export const NeutralMob: React.FC<NeutralMobProps> = ({
  position,
  modelName,
  scale = 1,
}) => {
  const groupRef = useRef<THREE.Group>(null);

  // Get the preloaded model from the asset preloader
  const model = useMemo(() => {
    const gltf = assetPreloader.getModel(modelName);
    if (!gltf) {
      console.warn(`[NeutralMob] Model not found in preloader: ${modelName}`);
      return null;
    }

    // Clone the model so each instance is independent
    const clonedScene = gltf.scene.clone();
    
    // Scale the model
    clonedScene.scale.set(scale, scale, scale);

    return clonedScene;
  }, [modelName, scale]);

  // Simple idle animation
  useFrame((state) => {
    if (groupRef.current) {
      // Gentle bobbing animation
      groupRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      
      // Slow rotation
      groupRef.current.rotation.y += 0.005;
    }
  });

  if (!model) {
    // Fallback: Show a placeholder cube if model isn't loaded
    return (
      <mesh position={position}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ff0000" />
      </mesh>
    );
  }

  return (
    <group ref={groupRef} position={position}>
      <primitive object={model} />
    </group>
  );
};

/**
 * Helper function to preload neutral mob models
 * Call this during initialization to register models for preloading
 */
export function registerNeutralMobAssets(modelUrls: { name: string; url: string }[]) {
  const assets = modelUrls.map((model) => ({
    name: model.name,
    url: model.url,
    type: 'model' as const,
    critical: false, // Neutral mobs are not critical for game start
  }));

  assetPreloader.registerAssets(assets);
}
