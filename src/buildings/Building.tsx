/**
 * Building Component with Ghost/Blueprint Preview
 * Handles rendering of buildings and their transparent preview during placement
 */

import React, { useMemo } from 'react';
import { BuildingType, getBuildingModel, BuildingPlacement } from './buildingModels';

export interface BuildingProps {
  type: BuildingType;
  position: { x: number; y: number; z: number };
  rotation?: number;
  isPlaced?: boolean; // true if building is placed, false if in preview mode
}

/**
 * BuildingGhost component - Renders transparent preview of building during placement
 * This is the component that was missing for archery range, blacksmith, and walls
 */
export const BuildingGhost: React.FC<BuildingProps> = ({
  type,
  position,
  rotation = 0,
  isPlaced = false,
}) => {
  const model = useMemo(() => getBuildingModel(type), [type]);

  if (!model.hasGhostPreview) {
    return null;
  }

  // Ghost preview styling - transparent and with visual indicators
  const ghostStyle: React.CSSProperties = {
    position: 'absolute',
    left: `${position.x}px`,
    top: `${position.z}px`,
    width: `${model.width * 32}px`, // 32px per grid unit
    height: `${model.depth * 32}px`,
    transform: `rotate(${rotation}deg)`,
    backgroundColor: model.color,
    opacity: isPlaced ? 1 : 0.5, // Transparent when not placed
    border: isPlaced ? 'none' : '2px dashed #fff',
    borderRadius: '4px',
    transition: 'opacity 0.2s ease',
    pointerEvents: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    color: '#fff',
    fontWeight: 'bold',
    boxShadow: isPlaced ? 'none' : '0 0 10px rgba(255, 255, 255, 0.3)',
  };

  return (
    <div 
      style={ghostStyle} 
      className="building-ghost"
      data-building-type={type}
    >
      {!isPlaced && <span>{model.name}</span>}
    </div>
  );
};

/**
 * Building Component - Renders a placed building
 */
export const Building: React.FC<BuildingProps> = (props) => {
  return <BuildingGhost {...props} isPlaced={true} />;
};

/**
 * BuildingPreview component - Shows preview during placement mode
 * This component specifically handles the blueprint/ghost preview
 */
export interface BuildingPreviewProps {
  placement: BuildingPlacement | null;
  onPlacementChange?: (placement: BuildingPlacement) => void;
}

export const BuildingPreview: React.FC<BuildingPreviewProps> = ({
  placement,
  onPlacementChange,
}) => {
  if (!placement) {
    return null;
  }

  const model = getBuildingModel(placement.type);

  // Add visual feedback for valid/invalid placement
  const previewStyle: React.CSSProperties = {
    filter: placement.isValid 
      ? 'brightness(1.2)' 
      : 'brightness(0.7) hue-rotate(120deg)', // Red tint for invalid placement
  };

  return (
    <div style={previewStyle}>
      <BuildingGhost
        type={placement.type}
        position={placement.position}
        rotation={placement.rotation}
        isPlaced={false}
      />
      {!placement.isValid && (
        <div
          style={{
            position: 'absolute',
            left: `${placement.position.x}px`,
            top: `${placement.position.z - 40}px`,
            color: '#ff4444',
            fontSize: '14px',
            fontWeight: 'bold',
            textShadow: '0 0 4px rgba(0, 0, 0, 0.8)',
          }}
        >
          ⚠ Invalid Placement
        </div>
      )}
    </div>
  );
};

/**
 * Hook for managing building placement
 */
export const useBuildingPlacement = (buildingType: BuildingType | null) => {
  const [placement, setPlacement] = React.useState<BuildingPlacement | null>(null);
  const [isPlacing, setIsPlacing] = React.useState(false);

  React.useEffect(() => {
    if (buildingType) {
      setIsPlacing(true);
      setPlacement({
        type: buildingType,
        position: { x: 0, y: 0, z: 0 },
        rotation: 0,
        isValid: true,
      });
    } else {
      setIsPlacing(false);
      setPlacement(null);
    }
  }, [buildingType]);

  const updatePlacement = React.useCallback((updates: Partial<BuildingPlacement>) => {
    setPlacement(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  const confirmPlacement = React.useCallback(() => {
    if (placement && placement.isValid) {
      setIsPlacing(false);
      return placement;
    }
    return null;
  }, [placement]);

  const cancelPlacement = React.useCallback(() => {
    setIsPlacing(false);
    setPlacement(null);
  }, []);

  return {
    placement,
    isPlacing,
    updatePlacement,
    confirmPlacement,
    cancelPlacement,
  };
};

export default Building;
