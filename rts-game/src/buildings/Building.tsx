import { useRef } from 'react'
import { Mesh } from 'three'

interface BuildingProps {
  position?: [number, number, number]
  scale?: number
  color?: string
}

/**
 * Building Component - Represents a building structure in the RTS game
 * 
 * Shadow Configuration:
 * - castShadow={true} on all meshes - buildings cast shadows on ground and other objects
 * - receiveShadow={true} on all meshes - buildings receive shadows from units and other buildings
 * 
 * This ensures all buildings consistently participate in the shadow system
 */
function Building({ position = [0, 0, 0], scale = 1.0, color = '#95a5a6' }: BuildingProps) {
  const mainRef = useRef<Mesh>(null)
  const roofRef = useRef<Mesh>(null)
  
  return (
    <group position={position} scale={scale}>
      {/* Main Building Structure */}
      <mesh 
        ref={mainRef}
        position={[0, 1, 0]}
        castShadow    // Building casts shadows
        receiveShadow // Building receives shadows
      >
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial 
          color={color}
          metalness={0.1}
          roughness={0.8}
        />
      </mesh>
      
      {/* Roof */}
      <mesh 
        ref={roofRef}
        position={[0, 2.3, 0]}
        rotation={[0, Math.PI / 4, 0]}
        castShadow    // Roof casts shadows
        receiveShadow // Roof receives shadows
      >
        <coneGeometry args={[1.6, 0.8, 4]} />
        <meshStandardMaterial 
          color="#e74c3c"
          metalness={0.2}
          roughness={0.7}
        />
      </mesh>
      
      {/* Door */}
      <mesh 
        position={[0, 0.5, 1.01]}
        castShadow    // Door casts shadows
        receiveShadow // Door receives shadows
      >
        <boxGeometry args={[0.6, 1, 0.1]} />
        <meshStandardMaterial 
          color="#8b4513"
          metalness={0.0}
          roughness={0.9}
        />
      </mesh>
      
      {/* Window 1 */}
      <mesh 
        position={[-0.6, 1.3, 1.01]}
        castShadow    // Windows cast shadows
        receiveShadow // Windows receive shadows
      >
        <boxGeometry args={[0.4, 0.4, 0.05]} />
        <meshStandardMaterial 
          color="#87ceeb"
          metalness={0.5}
          roughness={0.2}
          emissive="#87ceeb"
          emissiveIntensity={0.1}
        />
      </mesh>
      
      {/* Window 2 */}
      <mesh 
        position={[0.6, 1.3, 1.01]}
        castShadow    // Windows cast shadows
        receiveShadow // Windows receive shadows
      >
        <boxGeometry args={[0.4, 0.4, 0.05]} />
        <meshStandardMaterial 
          color="#87ceeb"
          metalness={0.5}
          roughness={0.2}
          emissive="#87ceeb"
          emissiveIntensity={0.1}
        />
      </mesh>
      
      {/* Foundation/Base */}
      <mesh 
        position={[0, 0.05, 0]}
        castShadow    // Foundation casts shadows
        receiveShadow // Foundation receives shadows
      >
        <boxGeometry args={[2.4, 0.1, 2.4]} />
        <meshStandardMaterial 
          color="#34495e"
          metalness={0.0}
          roughness={1.0}
        />
      </mesh>
    </group>
  )
}

export default Building
