import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stats } from '@react-three/drei'
import { Suspense } from 'react'
import RTSUnit from './units/RTSUnit'
import Building from './buildings/Building'
import './App.css'

/**
 * Main App Component - RTS Game with Proper Shadow Configuration
 * 
 * Shadow Configuration:
 * 1. Canvas shadows enabled
 * 2. DirectionalLight with shadow mapping configured
 * 3. All meshes have castShadow and receiveShadow enabled
 */
function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas
        shadows // Enable shadows in the renderer
        camera={{ position: [10, 10, 10], fov: 50 }}
        gl={{ 
          antialias: true,
          powerPreference: "high-performance"
        }}
      >
        {/* Ambient light for base illumination */}
        <ambientLight intensity={0.3} />
        
        {/* 
          DirectionalLight with shadows enabled
          - castShadow: true enables shadow casting
          - shadow-mapSize: Higher values = better quality shadows
          - shadow-camera: Defines the area that receives shadows
        */}
        <directionalLight
          position={[10, 20, 10]}
          intensity={1.5}
          castShadow
          shadow-mapSize={[2048, 2048]} // High quality shadow map
          shadow-camera-far={50}
          shadow-camera-left={-20}
          shadow-camera-right={20}
          shadow-camera-top={20}
          shadow-camera-bottom={-20}
          shadow-bias={-0.0001} // Reduces shadow acne
        />
        
        {/* Additional fill light (no shadows) */}
        <pointLight position={[-10, 10, -10]} intensity={0.5} />
        
        {/* Ground plane - receives shadows */}
        <mesh 
          rotation={[-Math.PI / 2, 0, 0]} 
          position={[0, 0, 0]}
          receiveShadow // Ground receives shadows
        >
          <planeGeometry args={[50, 50]} />
          <meshStandardMaterial color="#2d5016" />
        </mesh>
        
        <Suspense fallback={null}>
          {/* RTS Units - cast and receive shadows */}
          <RTSUnit position={[-3, 0, -3]} color="#ff6b6b" />
          <RTSUnit position={[0, 0, -3]} color="#4ecdc4" />
          <RTSUnit position={[3, 0, -3]} color="#45b7d1" />
          
          {/* Buildings - cast and receive shadows */}
          <Building position={[-5, 0, 2]} scale={1.5} color="#95a5a6" />
          <Building position={[3, 0, 3]} scale={1.2} color="#7f8c8d" />
          <Building position={[-2, 0, 5]} scale={1.0} color="#bdc3c7" />
        </Suspense>
        
        <OrbitControls 
          enableDamping
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={30}
        />
        
        <Stats />
      </Canvas>
      
      <div style={{
        position: 'absolute',
        top: 10,
        left: 10,
        color: 'white',
        fontFamily: 'monospace',
        background: 'rgba(0,0,0,0.7)',
        padding: '10px',
        borderRadius: '5px'
      }}>
        <h3 style={{ margin: 0 }}>Orca RTS - Shadow Demo</h3>
        <p style={{ margin: '5px 0', fontSize: '12px' }}>
          ✓ All models cast shadows<br/>
          ✓ All models receive shadows<br/>
          ✓ Shadow map: 2048x2048<br/>
          ✓ Proper light configuration
        </p>
      </div>
    </div>
  )
}

export default App
