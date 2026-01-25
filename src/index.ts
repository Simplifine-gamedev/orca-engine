// Main exports for the Wall Building System
export { WallSystem } from './buildings/WallSystem';
export type { WallSystemProps, WallSegment } from './buildings/WallSystem';

export { WallBuildPanel } from './ui/WallBuildPanel';
export type { WallBuildPanelProps } from './ui/WallBuildPanel';

export { WallSystemDemo } from './examples/WallSystemDemo';

export type {
  Position,
  TerrainTile,
  BuildingCosts,
  GameResources,
} from './buildings/types';
