import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function HeightmapTerrain() {
  const meshRef = useRef<THREE.Mesh>(null)

  // Generate heightmap data with varied terrain
  const { geometry, material } = useMemo(() => {
    const size = 100
    const segments = 128
    
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments)
    const positions = geometry.attributes.position.array as Float32Array
    
    // Create varied terrain with hills and valleys
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i]
      const y = positions[i + 1]
      
      // Multiple octaves of noise for natural terrain
      const height = 
        Math.sin(x * 0.1) * Math.cos(y * 0.1) * 1.5 +
        Math.sin(x * 0.3) * Math.cos(y * 0.3) * 0.5 +
        Math.sin(x * 0.05) * Math.cos(y * 0.05) * 2.0 +
        Math.random() * 0.1
      
      positions[i + 2] = height
    }
    
    geometry.computeVertexNormals()
    geometry.computeBoundingBox()

    // Create textured material with better colors
    const material = new THREE.MeshStandardMaterial({
      color: '#6b8e4a',
      roughness: 0.9,
      metalness: 0.1,
      flatShading: false,
      side: THREE.DoubleSide,
    })

    // Add vertex colors for terrain variety
    const colors = new Float32Array(positions.length)
    for (let i = 0; i < positions.length; i += 3) {
      const height = positions[i + 2]
      
      // Color based on height
      if (height > 2) {
        // Higher areas - rocky gray-brown
        colors[i] = 0.5 + Math.random() * 0.1
        colors[i + 1] = 0.45 + Math.random() * 0.1
        colors[i + 2] = 0.35 + Math.random() * 0.1
      } else if (height > 0.5) {
        // Mid areas - grassy green
        colors[i] = 0.35 + Math.random() * 0.15
        colors[i + 1] = 0.55 + Math.random() * 0.15
        colors[i + 2] = 0.25 + Math.random() * 0.1
      } else {
        // Lower areas - darker grass with dirt
        colors[i] = 0.3 + Math.random() * 0.1
        colors[i + 1] = 0.4 + Math.random() * 0.1
        colors[i + 2] = 0.2 + Math.random() * 0.1
      }
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    material.vertexColors = true

    return { geometry, material }
  }, [])

  return (
    <group>
      {/* Main terrain */}
      <mesh
        ref={meshRef}
        rotation={[-Math.PI / 2, 0, 0]}
        geometry={geometry}
        material={material}
        receiveShadow
        castShadow
      />
      
      {/* Ground plane for shadows */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.5, 0]}
        receiveShadow
      >
        <planeGeometry args={[120, 120]} />
        <meshStandardMaterial
          color="#4a5a3a"
          roughness={1}
          metalness={0}
        />
      </mesh>
      
      {/* Add grid lines for better depth perception */}
      <gridHelper
        args={[100, 50, '#88aa66', '#668844']}
        position={[0, -0.45, 0]}
        material-opacity={0.15}
        material-transparent={true}
      />
    </group>
  )
}
