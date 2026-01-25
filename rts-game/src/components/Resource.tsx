import { useGameStore, type Resource as ResourceType } from '../store/gameStore';

export function Resource({ resource }: { resource: ResourceType }) {
  const getColor = () => {
    switch (resource.type) {
      case 'gold': return '#FFD700';
      case 'wood': return '#8B4513';
      default: return '#808080';
    }
  };
  
  const getSize = () => {
    switch (resource.type) {
      case 'gold': return [2, 1.5, 2];
      case 'wood': return [1.5, 2, 1.5];
      default: return [1, 1, 1];
    }
  };
  
  const [width, height, depth] = getSize();
  
  return (
    <group position={[resource.position.x, resource.position.y, resource.position.z]}>
      {/* Resource base */}
      <mesh position={[0, height/2, 0]}>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial color={getColor()} metalness={0.6} roughness={0.4} />
      </mesh>
      
      {/* Label */}
      <mesh position={[0, height + 0.5, 0]}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshBasicMaterial color={getColor()} />
      </mesh>
    </group>
  );
}

export function Resources() {
  const resources = useGameStore((state) => state.resources);
  
  return (
    <>
      {resources.map((resource) => (
        <Resource key={resource.id} resource={resource} />
      ))}
    </>
  );
}
