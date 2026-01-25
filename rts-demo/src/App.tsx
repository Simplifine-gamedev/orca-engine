import { Canvas } from '@react-three/fiber'
import { OrbitControls, Sky, Stars } from '@react-three/drei'
import { HeightmapTerrain } from './terrain/HeightmapTerrain'
import { VegetationSystem } from './vegetation/VegetationSystem'
import { Unit } from './units/Unit'
import * as THREE from 'three'

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#87CEEB' }}>
      <Canvas
        shadows
        camera={{ position: [20, 15, 20], fov: 60 }}
        gl={{ 
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.2
        }}
      >
        {/* Improved Lighting Setup */}
        
        {/* High intensity ambient light to reduce darkness */}
        <ambientLight intensity={0.8} color="#e8f4ff" />
        
        {/* Hemisphere light for better ground illumination */}
        <hemisphereLight
          intensity={0.6}
          color="#ffffff"
          groundColor="#b97a56"
          position={[0, 50, 0]}
        />
        
        {/* Main directional light (sun) with enhanced intensity */}
        <directionalLight
          position={[50, 50, 25]}
          intensity={1.5}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-50}
          shadow-camera-right={50}
          shadow-camera-top={50}
          shadow-camera-bottom={-50}
          shadow-camera-near={0.5}
          shadow-camera-far={200}
          shadow-bias={-0.0001}
          color="#fff5e6"
        />
        
        {/* Fill light to reduce harsh shadows */}
        <directionalLight
          position={[-30, 30, -30]}
          intensity={0.4}
          color="#b0d4ff"
        />
        
        {/* Rim light for unit definition */}
        <directionalLight
          position={[0, 20, -50]}
          intensity={0.3}
          color="#ffd4a3"
        />

        {/* Sky and atmosphere */}
        <Sky
          distance={450000}
          sunPosition={[50, 50, 25]}
          inclination={0.6}
          azimuth={0.25}
          turbidity={2}
          rayleigh={1}
        />
        
        {/* Subtle stars for depth */}
        <Stars
          radius={300}
          depth={60}
          count={1000}
          factor={3}
          saturation={0}
          fade
          speed={0.5}
        />

        {/* Terrain with improved textures */}
        <HeightmapTerrain />

        {/* Vegetation and environmental objects */}
        <VegetationSystem />

        {/* Sample units with improved visibility */}
        <Unit position={[0, 0, 0]} color="#ff4444" />
        <Unit position={[5, 0, 5]} color="#4488ff" />
        <Unit position={[-5, 0, -5]} color="#44ff44" />
        <Unit position={[8, 0, -3]} color="#ffaa44" />

        {/* Camera controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={10}
          maxDistance={100}
          maxPolarAngle={Math.PI / 2.2}
        />
      </Canvas>
      
      {/* UI Overlay */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        color: 'white',
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '15px',
        borderRadius: '8px',
        fontFamily: 'monospace',
        fontSize: '14px',
        pointerEvents: 'none'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>Orca RTS Demo</div>
        <div style={{ fontSize: '12px', opacity: 0.9 }}>
          <div>🎮 Controls:</div>
          <div>• Left Click + Drag: Rotate</div>
          <div>• Right Click + Drag: Pan</div>
          <div>• Scroll: Zoom</div>
        </div>
      </div>
    </div>
  )
}

export default App
