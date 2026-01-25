import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { RTSUnit, GroupDestinationMarker } from '../units/RTSUnit';
import { PathVisibilitySettings } from './PathVisibilitySettings';
import { useGameStore } from '../store/gameStore';
import { useGroupDestination } from '../hooks/useGroupDestination';

export const GameScene: React.FC = () => {
  const units = useGameStore(state => state.units);
  const { groupDestination, selectedUnitCount, shouldShowGroupMarker } = useGroupDestination();
  
  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      {/* 3D Game View */}
      <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        
        {/* Ground grid */}
        <Grid 
          args={[50, 50]} 
          cellSize={1} 
          cellThickness={0.5} 
          cellColor="#6f6f6f" 
          sectionSize={5} 
          sectionThickness={1} 
          sectionColor="#4ea7fc"
          fadeDistance={50}
          fadeStrength={1}
        />
        
        {/* Render all units */}
        {units.map(unit => (
          <RTSUnit key={unit.id} unit={unit} />
        ))}
        
        {/* Show group destination marker if applicable */}
        {shouldShowGroupMarker && groupDestination && (
          <GroupDestinationMarker 
            position={groupDestination} 
            unitCount={selectedUnitCount} 
          />
        )}
        
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI / 2}
        />
      </Canvas>
      
      {/* UI Overlay - Settings Panel */}
      <div style={{
        position: 'absolute',
        top: 20,
        right: 20,
        zIndex: 1000
      }}>
        <PathVisibilitySettings />
      </div>
      
      {/* Info Panel */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        left: 20,
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '12px',
        borderRadius: '4px',
        border: '1px solid #4ea7fc',
        zIndex: 1000
      }}>
        <div>Total Units: {units.length}</div>
        <div>Selected Units: {selectedUnitCount}</div>
        {shouldShowGroupMarker && (
          <div style={{ color: '#4ea7fc' }}>Group Marker Active</div>
        )}
      </div>
    </div>
  );
};
