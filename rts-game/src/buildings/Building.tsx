'use client';

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh } from 'three';
import { PlacedBuilding } from '../types/building';
import { useBuildingStore } from '../store/buildingStore';

interface BuildingProps {
  building: PlacedBuilding;
}

export function Building({ building }: BuildingProps) {
  const meshRef = useRef<Mesh>(null);
  const selectBuilding = useBuildingStore(state => state.selectBuilding);
  const isConstructed = building.constructionProgress >= 100;
  
  // Animate buildings under construction
  useFrame((state) => {
    if (meshRef.current && !isConstructed) {
      meshRef.current.position.y = Math.sin(state.clock.elapsedTime * 2) * 0.05;
    }
  });
  
  const handleClick = () => {
    selectBuilding(building.instanceId);
  };
  
  // Calculate color based on health and construction
  const getColor = () => {
    if (!isConstructed) return '#888888';
    const healthPercent = building.health / building.hitPoints;
    if (healthPercent > 0.7) return '#4a9eff';
    if (healthPercent > 0.3) return '#ffa500';
    return '#ff4444';
  };
  
  return (
    <group position={[building.position.x, 0, building.position.y]}>
      {/* Main building mesh - placeholder box */}
      <mesh
        ref={meshRef}
        onClick={handleClick}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[building.size.width, 2, building.size.height]} />
        <meshStandardMaterial 
          color={getColor()} 
          opacity={isConstructed ? 1 : 0.6}
          transparent={!isConstructed}
        />
      </mesh>
      
      {/* Foundation */}
      <mesh position={[0, -0.6, 0]} receiveShadow>
        <boxGeometry args={[building.size.width + 0.2, 0.2, building.size.height + 0.2]} />
        <meshStandardMaterial color="#2c2c2c" />
      </mesh>
      
      {/* Construction progress indicator */}
      {!isConstructed && (
        <mesh position={[0, 2, 0]}>
          <boxGeometry args={[building.size.width, 0.1, 0.2]} />
          <meshBasicMaterial color="#00ff00" />
          <mesh position={[(building.constructionProgress / 100 - 0.5) * building.size.width, 0, 0]}>
            <boxGeometry args={[(building.constructionProgress / 100) * building.size.width, 0.12, 0.22]} />
            <meshBasicMaterial color="#ffff00" />
          </mesh>
        </mesh>
      )}
      
      {/* Health bar */}
      {isConstructed && building.health < building.hitPoints && (
        <mesh position={[0, 2.5, 0]}>
          <boxGeometry args={[building.size.width, 0.1, 0.1]} />
          <meshBasicMaterial color="#ff0000" />
          <mesh position={[((building.health / building.hitPoints) - 0.5) * building.size.width, 0, 0]}>
            <boxGeometry args={[(building.health / building.hitPoints) * building.size.width, 0.12, 0.12]} />
            <meshBasicMaterial color="#00ff00" />
          </mesh>
        </mesh>
      )}
    </group>
  );
}

export function BuildingPreview({ type }: { type: string }) {
  return (
    <div className="building-preview w-16 h-16 bg-gray-700 rounded-md border-2 border-gray-600 flex items-center justify-center">
      <div className="text-xs text-center text-white">
        {type}
      </div>
    </div>
  );
}
