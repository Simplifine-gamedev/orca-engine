import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh } from 'three'

interface BuildingProps {
  position?: [number, number, number]
  scale?: number
  color?: string
}

export function Building({ position = [0, 0, 0], scale = 1, color = '#8b4513' }: BuildingProps) {
  const groupRef = useRef<Mesh>(null)
  const [hovered, setHovered] = useState(false)

  // Subtle hover animation
  useFrame((state) => {
    if (groupRef.current && hovered) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 2) * 0.05
    }
  })

  const buildingHeight = 2 * scale
  const buildingWidth = 1.5 * scale
  const roofHeight = 0.8 * scale

  return (
    <group
      ref={groupRef}
      position={position}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* Main building structure - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh position={[0, buildingHeight / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[buildingWidth, buildingHeight, buildingWidth]} />
        <meshStandardMaterial
          color={hovered ? '#a0522d' : color}
          metalness={0.2}
          roughness={0.8}
        />
      </mesh>

      {/* Roof - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh
        position={[0, buildingHeight + roofHeight / 2, 0]}
        castShadow
        receiveShadow
      >
        <coneGeometry args={[buildingWidth * 0.8, roofHeight, 4]} />
        <meshStandardMaterial
          color="#654321"
          metalness={0.1}
          roughness={0.9}
        />
      </mesh>

      {/* Door - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh
        position={[0, buildingHeight * 0.25, buildingWidth / 2 + 0.01]}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[buildingWidth * 0.3, buildingHeight * 0.5, 0.1]} />
        <meshStandardMaterial
          color="#3d2817"
          metalness={0.1}
          roughness={0.95}
        />
      </mesh>

      {/* Windows - IMPORTANT: castShadow and receiveShadow enabled */}
      {[-0.4, 0.4].map((xOffset, idx) => (
        <mesh
          key={`window-${idx}`}
          position={[xOffset * scale, buildingHeight * 0.6, buildingWidth / 2 + 0.01]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[0.3 * scale, 0.3 * scale, 0.05]} />
          <meshStandardMaterial
            color="#87ceeb"
            metalness={0.9}
            roughness={0.1}
            emissive="#4a90e2"
            emissiveIntensity={0.3}
          />
        </mesh>
      ))}

      {/* Chimney - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh
        position={[buildingWidth * 0.3, buildingHeight + roofHeight * 0.8, 0]}
        castShadow
        receiveShadow
      >
        <cylinderGeometry args={[0.15 * scale, 0.15 * scale, 0.6 * scale, 8]} />
        <meshStandardMaterial
          color="#5c4033"
          metalness={0.1}
          roughness={0.9}
        />
      </mesh>

      {/* Foundation - IMPORTANT: castShadow and receiveShadow enabled */}
      <mesh position={[0, 0.1, 0]} castShadow receiveShadow>
        <boxGeometry args={[buildingWidth * 1.2, 0.2, buildingWidth * 1.2]} />
        <meshStandardMaterial
          color="#696969"
          metalness={0.3}
          roughness={0.85}
        />
      </mesh>
    </group>
  )
}
