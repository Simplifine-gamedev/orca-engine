import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh } from 'three'

interface RTSUnitProps {
  position?: [number, number, number]
  color?: string
}

export function RTSUnit({ position = [0, 0, 0], color = '#4ecdc4' }: RTSUnitProps) {
  const meshRef = useRef<Mesh>(null)

  // Subtle animation
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y = 0.5 + Math.sin(state.clock.elapsedTime + position[0]) * 0.1
    }
  })

  return (
    <group position={position}>
      {/* Main body - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh ref={meshRef} position={[0, 0.5, 0]} castShadow receiveShadow>
        <capsuleGeometry args={[0.3, 0.6, 8, 16]} />
        <meshStandardMaterial
          color={color}
          metalness={0.3}
          roughness={0.6}
        />
      </mesh>

      {/* Head/turret - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh position={[0, 1.2, 0]} castShadow receiveShadow>
        <sphereGeometry args={[0.25, 16, 16]} />
        <meshStandardMaterial
          color={color}
          metalness={0.4}
          roughness={0.5}
        />
      </mesh>

      {/* Weapon/antenna - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh position={[0, 1.4, 0.3]} rotation={[Math.PI / 4, 0, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.05, 0.05, 0.5, 8]} />
        <meshStandardMaterial
          color="#333333"
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Base platform - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh position={[0, 0.1, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.4, 0.4, 0.2, 16]} />
        <meshStandardMaterial
          color="#555555"
          metalness={0.6}
          roughness={0.4}
        />
      </mesh>
    </group>
  )
}
