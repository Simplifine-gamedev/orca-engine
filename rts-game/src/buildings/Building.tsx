import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGameStore } from '../store/gameStore';
import type { Building as BuildingType } from '../store/gameStore';

// Building component for placed buildings
export function Building({ building }: { building: BuildingType }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  const getSize = () => {
    switch (building.type) {
      case 'townhall': return [4, 3, 4];
      case 'barracks': return [3, 2.5, 3];
      case 'farm': return [2.5, 2, 2.5];
      default: return [2, 2, 2];
    }
  };
  
  const getColor = () => {
    switch (building.type) {
      case 'townhall': return '#8B4513';
      case 'barracks': return '#696969';
      case 'farm': return '#DAA520';
      default: return '#888888';
    }
  };
  
  const [width, height, depth] = getSize();
  
  return (
    <mesh
      ref={meshRef}
      position={[building.position.x, building.position.y + height/2, building.position.z]}
    >
      <boxGeometry args={[width, height, depth]} />
      <meshStandardMaterial 
        color={getColor()} 
        opacity={building.isConstructed ? 1.0 : 0.7}
        transparent={!building.isConstructed}
      />
    </mesh>
  );
}

// BuildingGhost component for placement preview
// BUG WAS HERE: The component wasn't checking if placement was active independently
export function BuildingGhost() {
  const meshRef = useRef<THREE.Mesh>(null);
  const buildingPlacement = useGameStore((state) => state.buildingPlacement);
  const workers = useGameStore((state) => state.workers);
  
  // BUG: This was the problem - checking worker state when we shouldn't
  // const hasWorkerMining = workers.some((w) => w.state === 'mining');
  
  // Pulse animation for ghost
  useFrame((state) => {
    if (meshRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.1 + 0.9;
      meshRef.current.scale.set(pulse, pulse, pulse);
    }
  });
  
  // FIX: Only check if building placement is active and has a ghost position
  // The building ghost should show regardless of worker state
  if (!buildingPlacement.isActive || !buildingPlacement.ghostPosition || !buildingPlacement.type) {
    return null;
  }
  
  const getSize = () => {
    switch (buildingPlacement.type) {
      case 'townhall': return [4, 3, 4];
      case 'barracks': return [3, 2.5, 3];
      case 'farm': return [2.5, 2, 2.5];
      default: return [2, 2, 2];
    }
  };
  
  const getColor = () => {
    switch (buildingPlacement.type) {
      case 'townhall': return '#8B4513';
      case 'barracks': return '#00FF00';
      case 'farm': return '#DAA520';
      default: return '#00FF00';
    }
  };
  
  const [width, height, depth] = getSize();
  const { x, y, z } = buildingPlacement.ghostPosition;
  
  return (
    <mesh
      ref={meshRef}
      position={[x, y + height/2, z]}
    >
      <boxGeometry args={[width, height, depth]} />
      <meshStandardMaterial 
        color={getColor()}
        transparent
        opacity={0.5}
        wireframe
      />
    </mesh>
  );
}

// Buildings container component
export function Buildings() {
  const buildings = useGameStore((state) => state.buildings);
  
  return (
    <>
      {buildings.map((building) => (
        <Building key={building.id} building={building} />
      ))}
      <BuildingGhost />
    </>
  );
}
