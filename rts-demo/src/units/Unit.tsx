import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface UnitProps {
  position: [number, number, number]
  color: string
}

export function Unit({ position, color }: UnitProps) {
  const meshRef = useRef<THREE.Group>(null)
  const [hovered, setHovered] = useState(false)
  const [selected, setSelected] = useState(false)
  
  // Animate unit with subtle hover effect
  useFrame((state) => {
    if (meshRef.current) {
      // Subtle floating animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.05
      
      // Rotate when hovered
      if (hovered) {
        meshRef.current.rotation.y += 0.02
      }
    }
  })

  // Calculate terrain height at position
  const x = position[0]
  const z = position[2]
  const terrainHeight = 
    Math.sin(x * 0.1) * Math.cos(z * 0.1) * 1.5 +
    Math.sin(x * 0.3) * Math.cos(z * 0.3) * 0.5 +
    Math.sin(x * 0.05) * Math.cos(z * 0.05) * 2.0

  return (
    <group
      ref={meshRef}
      position={[position[0], terrainHeight + 0.5, position[2]]}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onClick={() => setSelected(!selected)}
    >
      {/* Main unit body - bright and visible */}
      <mesh castShadow receiveShadow>
        <cylinderGeometry args={[0.4, 0.5, 0.8, 8]} />
        <meshStandardMaterial
          color={color}
          roughness={0.3}
          metalness={0.4}
          emissive={color}
          emissiveIntensity={hovered ? 0.3 : 0.1}
        />
      </mesh>
      
      {/* Head/turret - contrasting color */}
      <mesh position={[0, 0.6, 0]} castShadow>
        <sphereGeometry args={[0.3, 8, 6]} />
        <meshStandardMaterial
          color={new THREE.Color(color).multiplyScalar(0.7)}
          roughness={0.4}
          metalness={0.5}
        />
      </mesh>
      
      {/* Weapon/detail - accent color */}
      <mesh position={[0, 0.6, 0.4]} castShadow>
        <boxGeometry args={[0.1, 0.1, 0.6]} />
        <meshStandardMaterial
          color="#333333"
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>
      
      {/* Outline ring for better visibility */}
      <mesh position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.5, 0.65, 16]} />
        <meshBasicMaterial
          color={selected ? '#ffff00' : hovered ? '#ffffff' : color}
          transparent
          opacity={selected ? 0.8 : hovered ? 0.6 : 0.3}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Shadow blob for grounding */}
      <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[0.5, 16]} />
        <meshBasicMaterial
          color="#000000"
          transparent
          opacity={0.3}
        />
      </mesh>
      
      {/* Selection indicator */}
      {selected && (
        <mesh position={[0, 1.2, 0]}>
          <coneGeometry args={[0.2, 0.3, 3]} />
          <meshBasicMaterial color="#ffff00" />
        </mesh>
      )}
      
      {/* Team indicator light */}
      <pointLight
        position={[0, 1, 0]}
        intensity={0.5}
        distance={3}
        color={color}
      />
    </group>
  )
}
