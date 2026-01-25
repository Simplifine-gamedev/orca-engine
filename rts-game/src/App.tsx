import { Canvas } from '@react-three/fiber'
import { OrbitControls, Sky } from '@react-three/drei'
import { RTSUnit } from './units/RTSUnit'
import { Building } from './buildings/Building'

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas
        shadows={{
          enabled: true,
          type: 'PCFSoftShadowMap', // Soft shadows for better quality
        }}
        camera={{
          position: [10, 10, 10],
          fov: 50,
        }}
      >
        {/* Lighting setup with proper shadow configuration */}
        <ambientLight intensity={0.3} />
        
        {/* Main directional light with shadows enabled */}
        <directionalLight
          position={[10, 15, 5]}
          intensity={1.2}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-far={50}
          shadow-camera-left={-15}
          shadow-camera-right={15}
          shadow-camera-top={15}
          shadow-camera-bottom={-15}
          shadow-bias={-0.0001}
        />

        {/* Secondary fill light (no shadows for performance) */}
        <directionalLight
          position={[-5, 8, -5]}
          intensity={0.4}
          castShadow={false}
        />

        {/* Ground plane to receive shadows */}
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0, 0]}
          receiveShadow
        >
          <planeGeometry args={[50, 50]} />
          <meshStandardMaterial color="#2d5016" />
        </mesh>

        {/* RTS Units - with shadows enabled */}
        <RTSUnit position={[-3, 0, 0]} color="#ff6b6b" />
        <RTSUnit position={[-1, 0, 0]} color="#4ecdc4" />
        <RTSUnit position={[1, 0, 0]} color="#45b7d1" />
        <RTSUnit position={[3, 0, 2]} color="#f9ca24" />

        {/* Buildings - with shadows enabled */}
        <Building position={[0, 0, -5]} scale={1.5} color="#8b4513" />
        <Building position={[5, 0, -5]} scale={1.2} color="#654321" />
        <Building position={[-5, 0, -5]} scale={1.8} color="#5d4e37" />

        {/* Sky */}
        <Sky
          distance={450000}
          sunPosition={[10, 15, 5]}
          inclination={0.6}
          azimuth={0.25}
        />

        {/* Camera controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI / 2.5}
          minDistance={5}
          maxDistance={30}
        />
      </Canvas>
    </div>
  )
}

export default App
