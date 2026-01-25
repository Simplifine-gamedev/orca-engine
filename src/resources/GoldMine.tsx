import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface GoldMineProps {
  position: [number, number, number];
  onMinimapRegister?: (position: [number, number], color: string, size: number) => void;
}

export const GoldMine: React.FC<GoldMineProps> = ({ position, onMinimapRegister }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const particlesRef = useRef<THREE.Points>(null);

  // Increased scale from default to make gold mines more visible
  const SCALE_MULTIPLIER = 2.5; // 2.5x larger than before
  const baseScale = 1.0 * SCALE_MULTIPLIER;

  // Register on minimap with prominent marker
  useMemo(() => {
    if (onMinimapRegister) {
      // Gold color with larger size for minimap visibility
      onMinimapRegister(
        [position[0], position[2]], 
        '#FFD700', // Bright gold color
        8 // Larger minimap marker size (increased from typical 4-5)
      );
    }
  }, [position, onMinimapRegister]);

  // Create particle system for visual prominence
  const particleGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const count = 100;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      // Random positions around the gold mine
      const angle = Math.random() * Math.PI * 2;
      const radius = 0.5 + Math.random() * 1.5;
      const height = Math.random() * 3;

      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = height;
      positions[i * 3 + 2] = Math.sin(angle) * radius;

      // Gold-colored particles
      colors[i * 3] = 1.0; // R
      colors[i * 3 + 1] = 0.84; // G
      colors[i * 3 + 2] = 0.0; // B

      sizes[i] = Math.random() * 0.2 + 0.1;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    return geometry;
  }, []);

  const particleMaterial = useMemo(() => {
    return new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
  }, []);

  // Animate glow and particles for visual prominence
  useFrame((state) => {
    if (glowRef.current) {
      // Pulsing glow effect
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.2 + 1;
      glowRef.current.scale.setScalar(pulse * baseScale * 1.2);
      
      // Fade glow in and out
      const material = glowRef.current.material as THREE.MeshBasicMaterial;
      material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.15;
    }

    if (particlesRef.current) {
      // Rotate particles slowly
      particlesRef.current.rotation.y += 0.005;
      
      // Animate particle positions (floating up)
      const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 1] += 0.01; // Move up
        
        // Reset particles that go too high
        if (positions[i + 1] > 3) {
          positions[i + 1] = 0;
        }
      }
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }

    if (meshRef.current) {
      // Gentle bobbing animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime) * 0.1;
    }
  });

  return (
    <group position={position}>
      {/* Main gold mine mesh */}
      <mesh ref={meshRef} scale={[baseScale, baseScale, baseScale]}>
        {/* Simple gold mine geometry - can be replaced with loaded model */}
        <cylinderGeometry args={[0.8, 1.2, 2, 8]} />
        <meshStandardMaterial 
          color="#DAA520"
          metalness={0.7}
          roughness={0.3}
          emissive="#FFD700"
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* Gold details on top */}
      <mesh position={[0, 1.2 * baseScale, 0]} scale={[baseScale * 0.6, baseScale * 0.6, baseScale * 0.6]}>
        <sphereGeometry args={[0.5, 8, 8]} />
        <meshStandardMaterial 
          color="#FFD700"
          metalness={0.9}
          roughness={0.1}
          emissive="#FFD700"
          emissiveIntensity={0.4}
        />
      </mesh>

      {/* Glowing aura for visibility */}
      <mesh ref={glowRef} scale={[baseScale * 1.5, baseScale * 1.5, baseScale * 1.5]}>
        <sphereGeometry args={[1.5, 16, 16]} />
        <meshBasicMaterial 
          color="#FFD700"
          transparent
          opacity={0.3}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Particle system */}
      <points ref={particlesRef} geometry={particleGeometry} material={particleMaterial} />

      {/* Additional bright marker for long-distance visibility */}
      <mesh position={[0, 3 * baseScale, 0]}>
        <sphereGeometry args={[0.3 * baseScale, 8, 8]} />
        <meshBasicMaterial 
          color="#FFFF00"
          transparent
          opacity={0.8}
        />
      </mesh>
    </group>
  );
};

export default GoldMine;
