/**
 * Building System - Main Export File
 * Exports all building-related components, types, and utilities
 */

// Core building components
export { Building, BuildingGhost, BuildingPreview, useBuildingPlacement } from './Building';

// Wall system components
export { WallSystem, WallGhost, WallPreview, useWallPlacement, isValidWallPlacement, connectWalls } from './WallSystem';

// Building models and types
export { 
  BuildingType, 
  getBuildingModel,
  BUILDING_MODELS,
  type BuildingModel,
  type BuildingPlacement,
} from './buildingModels';

// Demo/Test component
export { default as BuildingDemo } from './BuildingDemo';
