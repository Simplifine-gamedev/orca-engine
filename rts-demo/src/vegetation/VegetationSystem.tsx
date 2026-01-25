import React, { useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { Instance, Instances } from '@react-three/drei';

interface VegetationSystemProps {
	terrainSize?: number;
	density?: number;
	seed?: number;
}

// Simple seeded random number generator
class SeededRandom {
	private seed: number;

	constructor(seed: number) {
		this.seed = seed;
	}

	random() {
		const x = Math.sin(this.seed++) * 10000;
		return x - Math.floor(x);
	}

	range(min: number, max: number) {
		return min + this.random() * (max - min);
	}
}

export const VegetationSystem: React.FC<VegetationSystemProps> = ({
	terrainSize = 100,
	density = 1,
	seed = 42,
}) => {
	const vegetation = useMemo(() => {
		const rng = new SeededRandom(seed);
		const items: {
			type: string;
			position: [number, number, number];
			rotation: [number, number, number];
			scale: [number, number, number];
			color: string;
		}[] = [];

		// Calculate number of items based on density
		const rockCount = Math.floor(50 * density);
		const treeCount = Math.floor(80 * density);
		const bushCount = Math.floor(100 * density);
		const grassPatchCount = Math.floor(150 * density);
		const flowerCount = Math.floor(120 * density);
		const mushroomCount = Math.floor(60 * density);

		// Helper to get random position on terrain
		const getRandomPos = (): [number, number, number] => {
			const x = rng.range(-terrainSize / 2, terrainSize / 2);
			const z = rng.range(-terrainSize / 2, terrainSize / 2);
			// Simple height calculation (can be replaced with actual terrain sampling)
			const y = Math.sin(x * 0.1) * 2 + Math.cos(z * 0.1) * 1.5;
			return [x, y, z];
		};

		// Scatter rocks of various sizes
		for (let i = 0; i < rockCount; i++) {
			const scale = rng.range(0.5, 2.5);
			const position = getRandomPos();
			items.push({
				type: 'rock',
				position: [position[0], position[1], position[2]],
				rotation: [
					rng.range(0, Math.PI * 2),
					rng.range(0, Math.PI * 2),
					rng.range(0, Math.PI * 2),
				],
				scale: [
					scale * rng.range(0.8, 1.2),
					scale * rng.range(0.6, 1.0),
					scale * rng.range(0.8, 1.2),
				],
				color: `hsl(${rng.range(20, 40)}, ${rng.range(15, 25)}%, ${rng.range(30, 50)}%)`,
			});
		}

		// Add trees with variety
		for (let i = 0; i < treeCount; i++) {
			const treeType = rng.random() > 0.5 ? 'pine' : 'oak';
			const scale = rng.range(2, 5);
			const position = getRandomPos();
			items.push({
				type: treeType,
				position: [position[0], position[1] + scale / 2, position[2]],
				rotation: [0, rng.range(0, Math.PI * 2), 0],
				scale: [
					scale * rng.range(0.8, 1.2),
					scale,
					scale * rng.range(0.8, 1.2),
				],
				color: treeType === 'pine'
					? `hsl(${rng.range(110, 130)}, ${rng.range(40, 60)}%, ${rng.range(20, 35)}%)`
					: `hsl(${rng.range(90, 120)}, ${rng.range(45, 65)}%, ${rng.range(25, 40)}%)`,
			});
		}

		// Add bushes
		for (let i = 0; i < bushCount; i++) {
			const scale = rng.range(0.8, 1.5);
			const position = getRandomPos();
			items.push({
				type: 'bush',
				position: [position[0], position[1] + scale / 3, position[2]],
				rotation: [0, rng.range(0, Math.PI * 2), 0],
				scale: [
					scale * rng.range(0.9, 1.3),
					scale * rng.range(0.7, 1.0),
					scale * rng.range(0.9, 1.3),
				],
				color: `hsl(${rng.range(100, 140)}, ${rng.range(50, 70)}%, ${rng.range(30, 45)}%)`,
			});
		}

		// Add grass patches
		for (let i = 0; i < grassPatchCount; i++) {
			const scale = rng.range(0.5, 1.2);
			const position = getRandomPos();
			items.push({
				type: 'grass',
				position: [position[0], position[1] + 0.1, position[2]],
				rotation: [0, rng.range(0, Math.PI * 2), 0],
				scale: [scale * rng.range(0.8, 1.5), scale * 0.5, scale * rng.range(0.8, 1.5)],
				color: `hsl(${rng.range(80, 120)}, ${rng.range(60, 80)}%, ${rng.range(40, 55)}%)`,
			});
		}

		// Add flowers
		for (let i = 0; i < flowerCount; i++) {
			const scale = rng.range(0.3, 0.7);
			const position = getRandomPos();
			const hue = rng.random() > 0.5 ? rng.range(0, 40) : rng.range(280, 340);
			items.push({
				type: 'flower',
				position: [position[0], position[1] + scale / 2, position[2]],
				rotation: [0, rng.range(0, Math.PI * 2), 0],
				scale: [scale, scale * rng.range(1.2, 1.8), scale],
				color: `hsl(${hue}, ${rng.range(70, 90)}%, ${rng.range(50, 70)}%)`,
			});
		}

		// Add mushrooms
		for (let i = 0; i < mushroomCount; i++) {
			const scale = rng.range(0.4, 0.9);
			const position = getRandomPos();
			items.push({
				type: 'mushroom',
				position: [position[0], position[1] + scale / 3, position[2]],
				rotation: [
					rng.range(-0.2, 0.2),
					rng.range(0, Math.PI * 2),
					rng.range(-0.2, 0.2),
				],
				scale: [scale, scale * rng.range(0.8, 1.2), scale],
				color: `hsl(${rng.range(0, 20)}, ${rng.range(60, 80)}%, ${rng.range(55, 75)}%)`,
			});
		}

		return items;
	}, [terrainSize, density, seed]);

	// Group items by type for efficient instanced rendering
	const groupedVegetation = useMemo(() => {
		const groups: Record<string, typeof vegetation> = {};
		vegetation.forEach((item) => {
			if (!groups[item.type]) {
				groups[item.type] = [];
			}
			groups[item.type].push(item);
		});
		return groups;
	}, [vegetation]);

	return (
		<group name="vegetation-system">
			{/* Rocks */}
			{groupedVegetation.rock && (
				<Instances limit={groupedVegetation.rock.length}>
					<dodecahedronGeometry args={[1, 0]} />
					<meshStandardMaterial />
					{groupedVegetation.rock.map((item, i) => (
						<Instance
							key={`rock-${i}`}
							position={item.position}
							rotation={item.rotation}
							scale={item.scale}
							color={item.color}
						/>
					))}
				</Instances>
			)}

			{/* Pine Trees */}
			{groupedVegetation.pine && (
				<Instances limit={groupedVegetation.pine.length}>
					<coneGeometry args={[1, 2, 6]} />
					<meshStandardMaterial />
					{groupedVegetation.pine.map((item, i) => (
						<Instance
							key={`pine-${i}`}
							position={item.position}
							rotation={item.rotation}
							scale={item.scale}
							color={item.color}
						/>
					))}
				</Instances>
			)}

			{/* Oak Trees */}
			{groupedVegetation.oak && (
				<Instances limit={groupedVegetation.oak.length}>
					<sphereGeometry args={[1, 8, 6]} />
					<meshStandardMaterial />
					{groupedVegetation.oak.map((item, i) => (
						<Instance
							key={`oak-${i}`}
							position={item.position}
							rotation={item.rotation}
							scale={item.scale}
							color={item.color}
						/>
					))}
				</Instances>
			)}

			{/* Bushes */}
			{groupedVegetation.bush && (
				<Instances limit={groupedVegetation.bush.length}>
					<icosahedronGeometry args={[1, 0]} />
					<meshStandardMaterial />
					{groupedVegetation.bush.map((item, i) => (
						<Instance
							key={`bush-${i}`}
							position={item.position}
							rotation={item.rotation}
							scale={item.scale}
							color={item.color}
						/>
					))}
				</Instances>
			)}

			{/* Grass Patches */}
			{groupedVegetation.grass && (
				<Instances limit={groupedVegetation.grass.length}>
					<coneGeometry args={[1, 1.5, 3]} />
					<meshStandardMaterial />
					{groupedVegetation.grass.map((item, i) => (
						<Instance
							key={`grass-${i}`}
							position={item.position}
							rotation={item.rotation}
							scale={item.scale}
							color={item.color}
						/>
					))}
				</Instances>
			)}

			{/* Flowers */}
			{groupedVegetation.flower && (
				<Instances limit={groupedVegetation.flower.length}>
					<cylinderGeometry args={[0.3, 0.05, 1, 5]} />
					<meshStandardMaterial />
					{groupedVegetation.flower.map((item, i) => (
						<Instance
							key={`flower-${i}`}
							position={item.position}
							rotation={item.rotation}
							scale={item.scale}
							color={item.color}
						/>
					))}
				</Instances>
			)}

			{/* Mushrooms */}
			{groupedVegetation.mushroom && (
				<group>
					<Instances limit={groupedVegetation.mushroom.length}>
						<capsuleGeometry args={[0.5, 0.5, 4, 8]} />
						<meshStandardMaterial />
						{groupedVegetation.mushroom.map((item, i) => (
							<Instance
								key={`mushroom-${i}`}
								position={item.position}
								rotation={item.rotation}
								scale={item.scale}
								color={item.color}
							/>
						))}
					</Instances>
				</group>
			)}
		</group>
	);
};

// Animated grass component for wind effect (optional)
export const AnimatedGrass: React.FC<{ position: [number, number, number] }> = ({ position }) => {
	const meshRef = React.useRef<THREE.Mesh>(null);

	useFrame((state) => {
		if (meshRef.current) {
			meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 2 + position[0]) * 0.1;
		}
	});

	return (
		<mesh ref={meshRef} position={position}>
			<coneGeometry args={[0.5, 1.5, 3]} />
			<meshStandardMaterial color="#6B8E23" />
		</mesh>
	);
};

export default VegetationSystem;
