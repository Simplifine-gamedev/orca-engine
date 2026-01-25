import { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGameStore, type Worker as WorkerType } from '../store/gameStore';

export function Worker({ worker }: { worker: WorkerType }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const selectedWorker = useGameStore((state) => state.selectedWorker);
  
  const isSelected = selectedWorker === worker.id;
  
  // Simple animation based on state
  useFrame((state) => {
    if (meshRef.current && worker.state === 'mining') {
      // Bob up and down while mining
      const bob = Math.sin(state.clock.elapsedTime * 4) * 0.1;
      meshRef.current.position.y = worker.position.y + 0.5 + bob;
    }
  });
  
  const getColor = () => {
    switch (worker.state) {
      case 'mining': return '#FFD700';
      case 'building': return '#8B4513';
      case 'moving': return '#87CEEB';
      default: return '#4169E1';
    }
  };
  
  return (
    <group position={[worker.position.x, worker.position.y, worker.position.z]}>
      {/* Worker body */}
      <mesh ref={meshRef} position={[0, 0.5, 0]}>
        <capsuleGeometry args={[0.3, 0.6, 8, 16]} />
        <meshStandardMaterial color={getColor()} />
      </mesh>
      
      {/* Selection indicator */}
      {isSelected && (
        <mesh position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.5, 0.6, 32]} />
          <meshBasicMaterial color="#00FF00" transparent opacity={0.7} />
        </mesh>
      )}
      
      {/* State indicator */}
      <mesh position={[0, 1.5, 0]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshBasicMaterial color={getColor()} />
      </mesh>
    </group>
  );
}

export function Workers() {
  const workers = useGameStore((state) => state.workers);
  
  return (
    <>
      {workers.map((worker) => (
        <Worker key={worker.id} worker={worker} />
      ))}
    </>
  );
}
