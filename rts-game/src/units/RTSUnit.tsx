import React, { useEffect, useRef, useState } from 'react';
import { useGameStore, Unit } from '../store/gameStore';

interface RTSUnitProps {
  unit: Unit;
}

export const RTSUnit: React.FC<RTSUnitProps> = ({ unit }) => {
  const pathVisibilityMode = useGameStore(state => state.pathVisibilityMode);
  const showPathLines = useGameStore(state => state.showPathLines);
  const pathFadeDuration = useGameStore(state => state.pathFadeDuration);
  const pathOpacity = useGameStore(state => state.pathOpacity);
  
  const [currentOpacity, setCurrentOpacity] = useState(pathOpacity);
  const fadeTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Determine if this unit's path should be visible
  const shouldShowPath = (): boolean => {
    if (!showPathLines || !unit.path || unit.path.length < 2) {
      return false;
    }
    
    if (!unit.isSelected) {
      return false;
    }
    
    switch (pathVisibilityMode) {
      case 'none':
        return false;
      
      case 'all':
        return true;
      
      case 'lead-only':
        return unit.isLeadUnit === true;
      
      case 'group-marker':
        // Don't show individual paths in group marker mode
        return false;
      
      case 'fade-quick':
        return true; // Will be handled by opacity animation
      
      default:
        return true;
    }
  };
  
  // Handle quick fade animation
  useEffect(() => {
    if (pathVisibilityMode === 'fade-quick' && unit.destination) {
      // Start at full opacity
      setCurrentOpacity(pathOpacity);
      
      // Clear any existing timer
      if (fadeTimerRef.current) {
        clearTimeout(fadeTimerRef.current);
      }
      
      // Fade out over time
      const startTime = Date.now();
      const fadeInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / pathFadeDuration, 1);
        const newOpacity = pathOpacity * (1 - progress);
        
        setCurrentOpacity(newOpacity);
        
        if (progress >= 1) {
          clearInterval(fadeInterval);
        }
      }, 16); // ~60fps
      
      fadeTimerRef.current = fadeInterval as any;
      
      return () => {
        clearInterval(fadeInterval);
      };
    } else {
      setCurrentOpacity(pathOpacity);
    }
  }, [unit.destination, pathVisibilityMode, pathFadeDuration, pathOpacity]);
  
  const isPathVisible = shouldShowPath();
  const finalOpacity = pathVisibilityMode === 'fade-quick' ? currentOpacity : pathOpacity;
  
  return (
    <group position={[unit.position.x, unit.position.y, unit.position.z]}>
      {/* Unit mesh - simple cube for now */}
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial 
          color={unit.isSelected ? '#4ea7fc' : '#888888'}
          emissive={unit.isLeadUnit ? '#ff6600' : '#000000'}
          emissiveIntensity={unit.isLeadUnit ? 0.5 : 0}
        />
      </mesh>
      
      {/* Path line visualization */}
      {isPathVisible && unit.path && (
        <PathLine 
          path={unit.path} 
          opacity={finalOpacity}
          isLeadUnit={unit.isLeadUnit || false}
        />
      )}
      
      {/* Unit label/name */}
      {unit.isSelected && (
        <mesh position={[0, 2, 0]}>
          <sphereGeometry args={[0.2, 8, 8]} />
          <meshBasicMaterial 
            color={unit.isLeadUnit ? '#ff6600' : '#4ea7fc'} 
            transparent 
            opacity={0.8}
          />
        </mesh>
      )}
    </group>
  );
};

interface PathLineProps {
  path: Array<{ x: number; y: number; z: number }>;
  opacity: number;
  isLeadUnit: boolean;
}

const PathLine: React.FC<PathLineProps> = ({ path, opacity, isLeadUnit }) => {
  if (path.length < 2) return null;
  
  // Create line points
  const points: [number, number, number][] = path.map(p => [p.x, p.y, p.z]);
  
  return (
    <>
      {/* Main path line */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={points.length}
            array={new Float32Array(points.flat())}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial 
          color={isLeadUnit ? '#ff9933' : '#4ea7fc'}
          transparent
          opacity={opacity}
          linewidth={isLeadUnit ? 3 : 2}
        />
      </line>
      
      {/* Destination marker */}
      {path.length > 0 && (
        <mesh position={[path[path.length - 1].x, path[path.length - 1].y, path[path.length - 1].z]}>
          <ringGeometry args={[0.8, 1.2, 16]} />
          <meshBasicMaterial 
            color={isLeadUnit ? '#ff9933' : '#4ea7fc'}
            transparent
            opacity={opacity * 0.8}
            side={2} // DoubleSide
          />
        </mesh>
      )}
    </>
  );
};

interface GroupDestinationMarkerProps {
  position: { x: number; y: number; z: number };
  unitCount: number;
}

export const GroupDestinationMarker: React.FC<GroupDestinationMarkerProps> = ({ 
  position, 
  unitCount 
}) => {
  const [pulseScale, setPulseScale] = useState(1);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setPulseScale(s => (s === 1 ? 1.2 : 1));
    }, 500);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <group position={[position.x, position.y, position.z]}>
      {/* Large ring marker */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} scale={[pulseScale, pulseScale, 1]}>
        <ringGeometry args={[2, 2.5, 32]} />
        <meshBasicMaterial 
          color="#4ea7fc" 
          transparent 
          opacity={0.6}
          side={2}
        />
      </mesh>
      
      {/* Inner ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.5, 1.8, 32]} />
        <meshBasicMaterial 
          color="#66ccff" 
          transparent 
          opacity={0.8}
          side={2}
        />
      </mesh>
      
      {/* Unit count indicator */}
      <mesh position={[0, 0.5, 0]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial color="#4ea7fc" />
      </mesh>
      
      {/* Flag/pointer */}
      <mesh position={[0, 1.5, 0]}>
        <coneGeometry args={[0.4, 1, 4]} />
        <meshStandardMaterial color="#4ea7fc" emissive="#4ea7fc" emissiveIntensity={0.5} />
      </mesh>
    </group>
  );
};
