import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

interface HeightmapTerrainProps {
	size?: number;
	resolution?: number;
	heightScale?: number;
	seed?: number;
	showWireframe?: boolean;
}

// Simplex-like noise function for terrain generation
class TerrainGenerator {
	private seed: number;

	constructor(seed: number) {
		this.seed = seed;
	}

	// Simple hash function
	private hash(x: number, y: number): number {
		const h = Math.sin(x * 12.9898 + y * 78.233 + this.seed) * 43758.5453;
		return h - Math.floor(h);
	}

	// Smooth interpolation
	private smoothstep(t: number): number {
		return t * t * (3 - 2 * t);
	}

	// 2D noise
	noise2D(x: number, y: number): number {
		const xi = Math.floor(x);
		const yi = Math.floor(y);
		const xf = x - xi;
		const yf = y - yi;

		const a = this.hash(xi, yi);
		const b = this.hash(xi + 1, yi);
		const c = this.hash(xi, yi + 1);
		const d = this.hash(xi + 1, yi + 1);

		const u = this.smoothstep(xf);
		const v = this.smoothstep(yf);

		return (
			a * (1 - u) * (1 - v) +
			b * u * (1 - v) +
			c * (1 - u) * v +
			d * u * v
		);
	}

	// Fractal Brownian Motion for more realistic terrain
	fbm(x: number, y: number, octaves: number = 6): number {
		let value = 0;
		let amplitude = 1;
		let frequency = 1;
		let maxValue = 0;

		for (let i = 0; i < octaves; i++) {
			value += this.noise2D(x * frequency, y * frequency) * amplitude;
			maxValue += amplitude;
			amplitude *= 0.5;
			frequency *= 2;
		}

		return value / maxValue;
	}
}

export const HeightmapTerrain: React.FC<HeightmapTerrainProps> = ({
	size = 100,
	resolution = 128,
	heightScale = 10,
	seed = 42,
	showWireframe = false,
}) => {
	const meshRef = useRef<THREE.Mesh>(null);

	const { geometry, heightMap } = useMemo(() => {
		const generator = new TerrainGenerator(seed);
		const geo = new THREE.PlaneGeometry(size, size, resolution - 1, resolution - 1);
		geo.rotateX(-Math.PI / 2);

		const vertices = geo.attributes.position.array as Float32Array;
		const heightMapData: number[][] = [];

		// Generate height map
		for (let i = 0; i < resolution; i++) {
			heightMapData[i] = [];
			for (let j = 0; j < resolution; j++) {
				const idx = (i * resolution + j) * 3;
				const x = vertices[idx] / size;
				const z = vertices[idx + 2] / size;

				// Generate multiple terrain features
				let height = 0;

				// Base terrain with rolling hills
				height += generator.fbm(x * 3, z * 3, 4) * 0.5;

				// Add larger mountain features
				height += generator.fbm(x * 1, z * 1, 3) * 0.3;

				// Add small details
				height += generator.fbm(x * 8, z * 8, 2) * 0.1;

				// Create some cliff-like features
				const cliffNoise = generator.noise2D(x * 2 + 100, z * 2 + 100);
				if (cliffNoise > 0.6) {
					height += (cliffNoise - 0.6) * 2;
				}

				// Apply height scale
				height *= heightScale;

				// Store in height map for collision detection or other uses
				heightMapData[i][j] = height;

				// Update vertex
				vertices[idx + 1] = height;
			}
		}

		// Recompute normals for proper lighting
		geo.computeVertexNormals();

		// Add UV2 for lightmaps if needed
		geo.setAttribute('uv2', geo.attributes.uv);

		return { geometry: geo, heightMap: heightMapData };
	}, [size, resolution, heightScale, seed]);

	// Create texture based on height for more realistic terrain
	const terrainTexture = useMemo(() => {
		const canvas = document.createElement('canvas');
		const size = 512;
		canvas.width = size;
		canvas.height = size;
		const ctx = canvas.getContext('2d');
		if (!ctx) return null;

		const imageData = ctx.createImageData(size, size);
		const data = imageData.data;

		for (let i = 0; i < size; i++) {
			for (let j = 0; j < size; j++) {
				const idx = (i * size + j) * 4;
				const x = (i / size - 0.5) * 2;
				const y = (j / size - 0.5) * 2;
				
				const generator = new TerrainGenerator(seed);
				const noise = generator.fbm(x * 10, y * 10, 3);

				// Grass
				let r = 80 + noise * 40;
				let g = 120 + noise * 60;
				let b = 60 + noise * 30;

				// Add dirt patches
				if (noise > 0.6) {
					r = 130 + noise * 40;
					g = 110 + noise * 30;
					b = 70 + noise * 20;
				}

				// Add rocky areas
				if (noise < 0.3) {
					r = 100 + noise * 50;
					g = 100 + noise * 50;
					b = 100 + noise * 50;
				}

				data[idx] = r;
				data[idx + 1] = g;
				data[idx + 2] = b;
				data[idx + 3] = 255;
			}
		}

		ctx.putImageData(imageData, 0, 0);

		const texture = new THREE.CanvasTexture(canvas);
		texture.wrapS = THREE.RepeatWrapping;
		texture.wrapT = THREE.RepeatWrapping;
		texture.repeat.set(4, 4);
		return texture;
	}, [seed]);

	// Optional: Add subtle animation for water or wind effect
	useFrame((state) => {
		if (meshRef.current && meshRef.current.material instanceof THREE.Material) {
			// Could add water animation or wind effects here
		}
	});

	return (
		<group name="terrain">
			<mesh
				ref={meshRef}
				geometry={geometry}
				receiveShadow
				castShadow
			>
				<meshStandardMaterial
					map={terrainTexture}
					wireframe={showWireframe}
					roughness={0.9}
					metalness={0.1}
					color="#7CBA6C"
				/>
			</mesh>

			{/* Add water plane at low level */}
			<mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
				<planeGeometry args={[size * 1.2, size * 1.2]} />
				<meshStandardMaterial
					color="#4A90E2"
					transparent
					opacity={0.6}
					roughness={0.1}
					metalness={0.8}
				/>
			</mesh>

			{/* Add cliff faces for dramatic terrain */}
			<CliffFaces size={size} heightScale={heightScale} seed={seed} />
		</group>
	);
};

// Component for adding cliff-like rock faces
const CliffFaces: React.FC<{ size: number; heightScale: number; seed: number }> = ({
	size,
	heightScale,
	seed,
}) => {
	const cliffs = useMemo(() => {
		const generator = new TerrainGenerator(seed + 1000);
		const cliffData: Array<{
			position: [number, number, number];
			rotation: [number, number, number];
			scale: [number, number, number];
		}> = [];

		// Generate a few dramatic cliff faces
		for (let i = 0; i < 8; i++) {
			const angle = (i / 8) * Math.PI * 2;
			const radius = size * 0.35;
			const x = Math.cos(angle) * radius + generator.noise2D(i, 0) * 10;
			const z = Math.sin(angle) * radius + generator.noise2D(0, i) * 10;
			const y = generator.noise2D(x, z) * heightScale * 0.5;

			if (generator.noise2D(x * 0.1, z * 0.1) > 0.5) {
				cliffData.push({
					position: [x, y + heightScale * 0.5, z],
					rotation: [0, angle + Math.PI / 2, 0],
					scale: [
						8 + generator.noise2D(x, z) * 4,
						12 + generator.noise2D(z, x) * 6,
						2 + generator.noise2D(x + z, 0) * 1,
					],
				});
			}
		}

		return cliffData;
	}, [size, heightScale, seed]);

	return (
		<group name="cliff-faces">
			{cliffs.map((cliff, i) => (
				<mesh
					key={`cliff-${i}`}
					position={cliff.position}
					rotation={cliff.rotation}
					scale={cliff.scale}
					castShadow
					receiveShadow
				>
					<boxGeometry />
					<meshStandardMaterial
						color="#8B7D6B"
						roughness={0.9}
						metalness={0.1}
					/>
				</mesh>
			))}
		</group>
	);
};

// Utility function to get height at a specific position (for collision detection)
export const getTerrainHeight = (
	x: number,
	z: number,
	heightMap: number[][],
	size: number
): number => {
	const resolution = heightMap.length;
	const halfSize = size / 2;

	// Convert world position to heightmap coordinates
	const i = Math.floor(((z + halfSize) / size) * resolution);
	const j = Math.floor(((x + halfSize) / size) * resolution);

	// Bounds check
	if (i < 0 || i >= resolution || j < 0 || j >= resolution) {
		return 0;
	}

	return heightMap[i][j];
};

export default HeightmapTerrain;
