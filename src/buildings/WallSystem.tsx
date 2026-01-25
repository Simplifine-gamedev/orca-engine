/**
 * Wall System Component
 * Handles wall placement, connection, and blueprint preview
 * Fixed: Blueprint preview now properly shows for walls during placement
 */

import React, { useState, useCallback, useMemo } from 'react';
import { BuildingType, getBuildingModel, BuildingPlacement } from './buildingModels';
import { BuildingGhost } from './Building';

export interface WallSegment {
  id: string;
  position: { x: number; y: number; z: number };
  rotation: number;
  connectedTo: string[]; // IDs of connected wall segments
}

export interface WallSystemProps {
  segments: WallSegment[];
  isPlacingWall: boolean;
  previewSegment?: WallSegment | null;
  onSegmentPlace?: (segment: WallSegment) => void;
}

/**
 * WallGhost component - Specific ghost preview for walls
 * This ensures walls show their blueprint preview during placement
 */
export const WallGhost: React.FC<{
  segment: WallSegment;
  isPreview?: boolean;
  isValid?: boolean;
}> = ({ segment, isPreview = false, isValid = true }) => {
  const wallModel = getBuildingModel(BuildingType.WALL);

  const ghostStyle: React.CSSProperties = {
    position: 'absolute',
    left: `${segment.position.x}px`,
    top: `${segment.position.z}px`,
    width: `${wallModel.width * 32}px`,
    height: `${wallModel.depth * 32}px`,
    transform: `rotate(${segment.rotation}deg)`,
    backgroundColor: isValid ? wallModel.color : '#ff4444',
    opacity: isPreview ? 0.5 : 1,
    border: isPreview ? '2px dashed #fff' : '1px solid #333',
    transition: 'all 0.2s ease',
    pointerEvents: 'none',
    boxShadow: isPreview 
      ? '0 0 10px rgba(255, 255, 255, 0.5)' 
      : '0 2px 4px rgba(0, 0, 0, 0.3)',
  };

  return (
    <div
      style={ghostStyle}
      className="wall-ghost"
      data-wall-id={segment.id}
      data-is-preview={isPreview}
    />
  );
};

/**
 * WallPreview component - Shows wall blueprint during placement
 * FIX: This component ensures walls show proper ghost preview
 */
export const WallPreview: React.FC<{
  segment: WallSegment | null;
  isValid: boolean;
}> = ({ segment, isValid }) => {
  if (!segment) {
    return null;
  }

  return (
    <div className="wall-preview">
      <WallGhost segment={segment} isPreview={true} isValid={isValid} />
      {!isValid && (
        <div
          style={{
            position: 'absolute',
            left: `${segment.position.x}px`,
            top: `${segment.position.z - 30}px`,
            color: '#ff4444',
            fontSize: '12px',
            fontWeight: 'bold',
            textShadow: '0 0 4px rgba(0, 0, 0, 0.8)',
            whiteSpace: 'nowrap',
          }}
        >
          ⚠ Cannot place here
        </div>
      )}
    </div>
  );
};

/**
 * WallSystem component - Main component for wall management
 * Includes proper blueprint preview functionality
 */
export const WallSystem: React.FC<WallSystemProps> = ({
  segments,
  isPlacingWall,
  previewSegment,
  onSegmentPlace,
}) => {
  return (
    <div className="wall-system">
      {/* Render all placed wall segments */}
      {segments.map((segment) => (
        <WallGhost
          key={segment.id}
          segment={segment}
          isPreview={false}
          isValid={true}
        />
      ))}

      {/* Render preview segment during placement - FIX: Now properly shows */}
      {isPlacingWall && previewSegment && (
        <WallPreview segment={previewSegment} isValid={true} />
      )}
    </div>
  );
};

/**
 * Hook for managing wall placement with preview
 */
export const useWallPlacement = () => {
  const [segments, setSegments] = useState<WallSegment[]>([]);
  const [isPlacing, setIsPlacing] = useState(false);
  const [previewSegment, setPreviewSegment] = useState<WallSegment | null>(null);
  const [cursorPosition, setCursorPosition] = useState<{ x: number; z: number }>({ x: 0, z: 0 });

  const startPlacement = useCallback(() => {
    setIsPlacing(true);
  }, []);

  const updatePreview = useCallback((x: number, z: number) => {
    setCursorPosition({ x, z });
    
    // Snap to grid
    const gridSize = 32;
    const snappedX = Math.round(x / gridSize) * gridSize;
    const snappedZ = Math.round(z / gridSize) * gridSize;

    // Determine rotation based on nearby walls (simplified)
    const rotation = 0; // Could be enhanced with auto-rotation logic

    setPreviewSegment({
      id: `wall-preview-${Date.now()}`,
      position: { x: snappedX, y: 0, z: snappedZ },
      rotation,
      connectedTo: [],
    });
  }, []);

  const placeSegment = useCallback(() => {
    if (previewSegment && isPlacing) {
      const newSegment: WallSegment = {
        ...previewSegment,
        id: `wall-${Date.now()}`,
      };
      
      setSegments(prev => [...prev, newSegment]);
      
      // Keep placing mode active for continuous wall building
      // Preview will update on next mouse move
    }
  }, [previewSegment, isPlacing]);

  const cancelPlacement = useCallback(() => {
    setIsPlacing(false);
    setPreviewSegment(null);
  }, []);

  const clearWalls = useCallback(() => {
    setSegments([]);
  }, []);

  return {
    segments,
    isPlacing,
    previewSegment,
    startPlacement,
    updatePreview,
    placeSegment,
    cancelPlacement,
    clearWalls,
  };
};

/**
 * Utility function to check if wall placement is valid
 */
export const isValidWallPlacement = (
  segment: WallSegment,
  existingSegments: WallSegment[],
  minDistance: number = 32
): boolean => {
  // Check if too close to existing walls
  for (const existing of existingSegments) {
    const dx = segment.position.x - existing.position.x;
    const dz = segment.position.z - existing.position.z;
    const distance = Math.sqrt(dx * dx + dz * dz);
    
    if (distance < minDistance) {
      return false;
    }
  }
  
  return true;
};

/**
 * Utility function to auto-connect walls
 */
export const connectWalls = (segments: WallSegment[], connectionRadius: number = 48): WallSegment[] => {
  return segments.map(segment => {
    const connectedTo: string[] = [];
    
    for (const other of segments) {
      if (segment.id === other.id) continue;
      
      const dx = segment.position.x - other.position.x;
      const dz = segment.position.z - other.position.z;
      const distance = Math.sqrt(dx * dx + dz * dz);
      
      if (distance <= connectionRadius) {
        connectedTo.push(other.id);
      }
    }
    
    return { ...segment, connectedTo };
  });
};

export default WallSystem;
