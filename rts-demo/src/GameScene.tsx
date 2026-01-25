import React, { Suspense, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, Sky, Stats } from '@react-three/drei';
import { VegetationSystem } from './vegetation/VegetationSystem';
import { HeightmapTerrain } from './terrain/HeightmapTerrain';

interface GameSceneProps {
	debug?: boolean;
}

export const GameScene: React.FC<GameSceneProps> = ({ debug = false }) => {
	const [vegetationDensity, setVegetationDensity] = useState(1.0);
	const [showWireframe, setShowWireframe] = useState(false);

	return (
		<div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
			{/* Control Panel */}
			<div
				style={{
					position: 'absolute',
					top: 20,
					left: 20,
					zIndex: 1000,
					background: 'rgba(0, 0, 0, 0.7)',
					color: 'white',
					padding: '20px',
					borderRadius: '8px',
					fontFamily: 'monospace',
					maxWidth: '300px',
				}}
			>
				<h2 style={{ margin: '0 0 10px 0', fontSize: '18px' }}>Orca RTS - Map Controls</h2>
				
				<div style={{ marginBottom: '15px' }}>
					<label style={{ display: 'block', marginBottom: '5px' }}>
						Vegetation Density: {vegetationDensity.toFixed(1)}
					</label>
					<input
						type="range"
						min="0.1"
						max="2"
						step="0.1"
						value={vegetationDensity}
						onChange={(e) => setVegetationDensity(parseFloat(e.target.value))}
						style={{ width: '100%' }}
					/>
				</div>

				<div style={{ marginBottom: '15px' }}>
					<label style={{ display: 'flex', alignItems: 'center' }}>
						<input
							type="checkbox"
							checked={showWireframe}
							onChange={(e) => setShowWireframe(e.target.checked)}
							style={{ marginRight: '8px' }}
						/>
						Show Wireframe
					</label>
				</div>

				<div style={{ fontSize: '12px', opacity: 0.8, marginTop: '15px' }}>
					<p style={{ margin: '5px 0' }}>🌳 Trees: Pine & Oak varieties</p>
					<p style={{ margin: '5px 0' }}>🪨 Rocks: Various sizes</p>
					<p style={{ margin: '5px 0' }}>🌿 Bushes & Grass patches</p>
					<p style={{ margin: '5px 0' }}>🌸 Flowers & Mushrooms</p>
					<p style={{ margin: '5px 0' }}>🏔️ Hills & Cliff features</p>
				</div>

				<div style={{ fontSize: '11px', opacity: 0.6, marginTop: '15px', borderTop: '1px solid #444', paddingTop: '10px' }}>
					Controls: Mouse to orbit, scroll to zoom
				</div>
			</div>

			{/* 3D Canvas */}
			<Canvas
				shadows
				camera={{
					position: [50, 40, 50],
					fov: 60,
					near: 0.1,
					far: 1000,
				}}
			>
				{/* Lighting */}
				<ambientLight intensity={0.3} />
				<directionalLight
					position={[50, 50, 25]}
					intensity={0.8}
					castShadow
					shadow-mapSize={[2048, 2048]}
					shadow-camera-left={-60}
					shadow-camera-right={60}
					shadow-camera-top={60}
					shadow-camera-bottom={-60}
				/>
				<pointLight position={[-30, 20, -30]} intensity={0.3} color="#FFA500" />

				{/* Sky */}
				<Sky
					distance={450000}
					sunPosition={[100, 20, 100]}
					inclination={0.6}
					azimuth={0.25}
				/>

				{/* Environment for reflections */}
				<Environment preset="sunset" />

				<Suspense
					fallback={
						<mesh>
							<boxGeometry />
							<meshBasicMaterial color="hotpink" wireframe />
						</mesh>
					}
				>
					{/* Terrain */}
					<HeightmapTerrain
						size={100}
						resolution={128}
						heightScale={10}
						seed={42}
						showWireframe={showWireframe}
					/>

					{/* Vegetation */}
					<VegetationSystem
						terrainSize={100}
						density={vegetationDensity}
						seed={42}
					/>
				</Suspense>

				{/* Camera Controls */}
				<OrbitControls
					enablePan
					enableZoom
					enableRotate
					minDistance={10}
					maxDistance={200}
					maxPolarAngle={Math.PI / 2 - 0.1}
				/>

				{/* Debug Stats */}
				{debug && <Stats />}
			</Canvas>
		</div>
	);
};

export default GameScene;
