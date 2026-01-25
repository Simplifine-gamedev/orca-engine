/**
 * Orca RTS - Economy & Research System
 * Main export file for all stores and UI components
 */

// Stores
export { 
  gameStore,
  BuildingType,
  UnitType,
  BUILDING_UPGRADES,
  UNIT_UPGRADES,
  type GameState,
  type Resources,
  type Building,
  type Unit,
  type BuildingUpgrade,
  type UnitUpgrade,
} from './store/gameStore';

export {
  researchStore,
  ResearchCategory,
  RESEARCH_TREE,
  type Research,
  type ResearchEffect,
  type ActiveResearch,
} from './store/researchStore';

// UI Components
export { default as ResearchPanel } from './ui/ResearchPanel';
export { default as BuildingUpgradePanel } from './ui/BuildingUpgradePanel';
export { default as UnitUpgradePanel } from './ui/UnitUpgradePanel';
