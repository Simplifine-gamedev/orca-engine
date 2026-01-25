import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { assetPreloader } from '../systems/AssetPreloader';

interface RTSUnitProps {
  position: [number, number, number];
  modelName: string;
  scale?: number;
  team?: 'blue' | 'red';
  unitType?: 'worker' | 'soldier' | 'builder';
  onSelect?: (unit: RTSUnitProps) => void;
}

/**
 * RTSUnit - Controllable RTS unit
 * 
 * This component represents a player-controlled unit.
 * Models are preloaded during the loading screen to avoid lag during gameplay.
 * 
 * Fix for: "When game starts, some of the models aren't already loaded"
 * 
 * Features:
 * - Uses preloaded models for instant rendering
 * - Color-coded by team
 * - Different behavior based on unit type
 * - Clickable for selection
 */
export const RTSUnit: React.FC<RTSUnitProps> = ({
  position,
  modelName,
  scale = 1,
  team = 'blue',
  unitType = 'worker',
  onSelect,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const [isSelected, setIsSelected] = useState(false);
  const [model, setModel] = useState<THREE.Object3D | null>(null);

  // Load the preloaded model
  useEffect(() => {
    const gltf = assetPreloader.getModel(modelName);
    if (!gltf) {
      console.warn(`[RTSUnit] Model not found in preloader: ${modelName}`);
      return;
    }

    // Clone the model for this instance
    const clonedScene = gltf.scene.clone();
    clonedScene.scale.set(scale, scale, scale);

    // Apply team color tint to materials
    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        if (mesh.material) {
          const material = (mesh.material as THREE.MeshStandardMaterial).clone();
          
          // Tint the material based on team
          const teamColor = team === 'blue' ? new THREE.Color(0.3, 0.5, 1.0) : new THREE.Color(1.0, 0.3, 0.3);
          material.color.multiply(teamColor);
          
          mesh.material = material;
        }
      }
    });

    setModel(clonedScene);
  }, [modelName, scale, team]);

  // Unit-specific animations
  useFrame((state) => {
    if (groupRef.current) {
      switch (unitType) {
        case 'worker':
          // Workers bob up and down while working
          groupRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 3) * 0.05;
          break;
        case 'soldier':
          // Soldiers stand still but alert
          groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
          break;
        case 'builder':
          // Builders rotate slowly
          groupRef.current.rotation.y += 0.01;
          break;
      }

      // Selection highlight
      if (isSelected) {
        groupRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 4) * 0.05);
      }
    }
  });

  // Handle click for selection
  const handleClick = (event: THREE.Event) => {
    event.stopPropagation();
    setIsSelected(!isSelected);
    if (onSelect) {
      onSelect({ position, modelName, scale, team, unitType, onSelect });
    }
  };

  if (!model) {
    // Fallback: Show a colored box if model isn't loaded
    const fallbackColor = team === 'blue' ? '#3366ff' : '#ff3333';
    return (
      <mesh position={position} onClick={handleClick as any}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color={fallbackColor} />
      </mesh>
    );
  }

  return (
    <group ref={groupRef} position={position} onClick={handleClick as any}>
      <primitive object={model} />
      
      {/* Selection indicator */}
      {isSelected && (
        <mesh position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.8, 1, 32]} />
          <meshBasicMaterial color={team === 'blue' ? '#00ffff' : '#ffff00'} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
};

/**
 * Helper function to preload RTS unit models
 * Call this during initialization to register models for preloading
 */
export function registerRTSUnitAssets(modelUrls: { name: string; url: string; critical?: boolean }[]) {
  const assets = modelUrls.map((model) => ({
    name: model.name,
    url: model.url,
    type: 'model' as const,
    critical: model.critical !== undefined ? model.critical : true, // RTS units are critical by default
  }));

  assetPreloader.registerAssets(assets);
}
