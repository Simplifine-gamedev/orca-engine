import { useMemo } from 'react'

// Rock component
function Rock({ position, scale = 1 }: { position: [number, number, number], scale?: number }) {
  const colors = ['#888888', '#999999', '#777777', '#aaaaaa']
  const color = colors[Math.floor(Math.random() * colors.length)]
  
  return (
    <mesh position={position} castShadow receiveShadow>
      <dodecahedronGeometry args={[scale, 0]} />
      <meshStandardMaterial
        color={color}
        roughness={0.95}
        metalness={0.1}
      />
    </mesh>
  )
}

// Tree component
function Tree({ position }: { position: [number, number, number] }) {
  const trunkHeight = 1.5 + Math.random() * 1
  const crownSize = 0.8 + Math.random() * 0.5
  
  return (
    <group position={position}>
      {/* Trunk */}
      <mesh position={[0, trunkHeight / 2, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.15, 0.2, trunkHeight, 8]} />
        <meshStandardMaterial color="#654321" roughness={0.9} />
      </mesh>
      
      {/* Crown - multiple layers */}
      <mesh position={[0, trunkHeight + crownSize * 0.3, 0]} castShadow receiveShadow>
        <coneGeometry args={[crownSize, crownSize * 1.5, 8]} />
        <meshStandardMaterial color="#2d5016" roughness={0.8} />
      </mesh>
      <mesh position={[0, trunkHeight + crownSize * 0.8, 0]} castShadow receiveShadow>
        <coneGeometry args={[crownSize * 0.75, crownSize * 1.2, 8]} />
        <meshStandardMaterial color="#3a6b1f" roughness={0.8} />
      </mesh>
      <mesh position={[0, trunkHeight + crownSize * 1.2, 0]} castShadow receiveShadow>
        <coneGeometry args={[crownSize * 0.5, crownSize * 0.9, 8]} />
        <meshStandardMaterial color="#458622" roughness={0.8} />
      </mesh>
    </group>
  )
}

// Bush component
function Bush({ position, scale = 1 }: { position: [number, number, number], scale?: number }) {
  return (
    <group position={position}>
      <mesh position={[0, scale * 0.3, 0]} castShadow receiveShadow>
        <sphereGeometry args={[scale * 0.4, 8, 6]} />
        <meshStandardMaterial color="#4a7c2d" roughness={0.9} />
      </mesh>
      <mesh position={[scale * 0.2, scale * 0.25, scale * 0.1]} castShadow receiveShadow>
        <sphereGeometry args={[scale * 0.3, 8, 6]} />
        <meshStandardMaterial color="#567d35" roughness={0.9} />
      </mesh>
      <mesh position={[-scale * 0.15, scale * 0.2, -scale * 0.1]} castShadow receiveShadow>
        <sphereGeometry args={[scale * 0.25, 8, 6]} />
        <meshStandardMaterial color="#3d6b24" roughness={0.9} />
      </mesh>
    </group>
  )
}

// Grass patch component
function GrassPatch({ position }: { position: [number, number, number] }) {
  const grassBlades = useMemo(() => {
    const blades = []
    for (let i = 0; i < 15; i++) {
      const x = (Math.random() - 0.5) * 0.5
      const z = (Math.random() - 0.5) * 0.5
      const height = 0.3 + Math.random() * 0.2
      const rotation = Math.random() * Math.PI * 2
      blades.push({ x, z, height, rotation })
    }
    return blades
  }, [])

  return (
    <group position={position}>
      {grassBlades.map((blade, i) => (
        <mesh
          key={i}
          position={[blade.x, blade.height / 2, blade.z]}
          rotation={[0, blade.rotation, 0]}
        >
          <boxGeometry args={[0.02, blade.height, 0.01]} />
          <meshStandardMaterial color="#5a8a3a" roughness={0.9} />
        </mesh>
      ))}
    </group>
  )
}

export function VegetationSystem() {
  // Generate random positions for vegetation
  const vegetation = useMemo(() => {
    const items = []
    const spread = 40
    
    // Add trees
    for (let i = 0; i < 25; i++) {
      const x = (Math.random() - 0.5) * spread
      const z = (Math.random() - 0.5) * spread
      const y = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 1.5 +
               Math.sin(x * 0.05) * Math.cos(z * 0.05) * 2.0
      items.push({ type: 'tree', position: [x, y, z] as [number, number, number] })
    }
    
    // Add rocks
    for (let i = 0; i < 40; i++) {
      const x = (Math.random() - 0.5) * spread
      const z = (Math.random() - 0.5) * spread
      const y = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 1.5 +
               Math.sin(x * 0.05) * Math.cos(z * 0.05) * 2.0
      const scale = 0.3 + Math.random() * 0.7
      items.push({ type: 'rock', position: [x, y, z] as [number, number, number], scale })
    }
    
    // Add bushes
    for (let i = 0; i < 30; i++) {
      const x = (Math.random() - 0.5) * spread
      const z = (Math.random() - 0.5) * spread
      const y = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 1.5 +
               Math.sin(x * 0.05) * Math.cos(z * 0.05) * 2.0
      const scale = 0.5 + Math.random() * 0.5
      items.push({ type: 'bush', position: [x, y, z] as [number, number, number], scale })
    }
    
    // Add grass patches
    for (let i = 0; i < 50; i++) {
      const x = (Math.random() - 0.5) * spread
      const z = (Math.random() - 0.5) * spread
      const y = Math.sin(x * 0.1) * Math.cos(z * 0.1) * 1.5 +
               Math.sin(x * 0.05) * Math.cos(z * 0.05) * 2.0
      items.push({ type: 'grass', position: [x, y, z] as [number, number, number] })
    }
    
    return items
  }, [])

  return (
    <group>
      {vegetation.map((item, i) => {
        switch (item.type) {
          case 'tree':
            return <Tree key={`tree-${i}`} position={item.position} />
          case 'rock':
            return <Rock key={`rock-${i}`} position={item.position} scale={item.scale} />
          case 'bush':
            return <Bush key={`bush-${i}`} position={item.position} scale={item.scale} />
          case 'grass':
            return <GrassPatch key={`grass-${i}`} position={item.position} />
          default:
            return null
        }
      })}
    </group>
  )
}
