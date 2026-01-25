import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh } from 'three'

interface RTSUnitProps {
  position?: [number, number, number]
  color?: string
}

/**
 * RTSUnit Component - Represents a unit in the RTS game
 * 
 * Shadow Configuration:
 * - castShadow={true} on all meshes - units cast shadows
 * - receiveShadow={true} on all meshes - units can receive shadows from other objects
 * 
 * This ensures all units consistently cast and receive shadows
 */
function RTSUnit({ position = [0, 0, 0], color = '#ff6b6b' }: RTSUnitProps) {
  const bodyRef = useRef<Mesh>(null)
  const turretRef = useRef<Mesh>(null)
  
  // Simple animation - turret rotation
  useFrame((state) => {
    if (turretRef.current) {
      turretRef.current.rotation.y = Math.sin(state.clock.elapsedTime) * 0.3
    }
  })
  
  return (
    <group position={position}>
      {/* Unit Body - Main hull */}
      <mesh 
        ref={bodyRef}
        position={[0, 0.3, 0]}
        castShadow    // This mesh casts shadows
        receiveShadow // This mesh receives shadows
      >
        <boxGeometry args={[0.8, 0.4, 1.0]} />
        <meshStandardMaterial 
          color={color}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>
      
      {/* Unit Turret */}
      <mesh 
        ref={turretRef}
        position={[0, 0.6, 0]}
        castShadow    // Turret casts shadows
        receiveShadow // Turret receives shadows
      >
        <cylinderGeometry args={[0.25, 0.3, 0.3, 8]} />
        <meshStandardMaterial 
          color={color}
          metalness={0.4}
          roughness={0.6}
        />
      </mesh>
      
      {/* Turret Barrel */}
      <mesh 
        position={[0, 0.6, 0.4]}
        rotation={[Math.PI / 2, 0, 0]}
        castShadow    // Barrel casts shadows
        receiveShadow // Barrel receives shadows
      >
        <cylinderGeometry args={[0.08, 0.08, 0.6, 8]} />
        <meshStandardMaterial 
          color="#2c3e50"
          metalness={0.8}
          roughness={0.3}
        />
      </mesh>
      
      {/* Selection Ring (visual indicator) */}
      <mesh 
        position={[0, 0.02, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        castShadow={false}    // Selection ring doesn't need to cast shadows
        receiveShadow={false} // Selection ring doesn't need to receive shadows
      >
        <ringGeometry args={[0.6, 0.7, 32]} />
        <meshBasicMaterial color={color} opacity={0.3} transparent />
      </mesh>
    </group>
  )
}

export default RTSUnit
