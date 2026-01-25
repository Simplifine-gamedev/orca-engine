import { create } from 'zustand';
import { Building, Unit, Resource, Position, RallyPoint, GameState } from '../types';

interface GameStore extends GameState {
  setRallyPoint: (buildingId: string, position: Position) => void;
  spawnUnit: (buildingId: string, unitType: string) => void;
  detectResourceAtPosition: (position: Position) => Resource | null;
  assignUnitToGather: (unitId: string, resourceId: string) => void;
  addBuilding: (building: Building) => void;
  addResource: (resource: Resource) => void;
}

const RESOURCE_DETECTION_RADIUS = 50;

export const useGameStore = create<GameStore>((set, get) => ({
  buildings: new Map(),
  units: new Map(),
  resources: new Map(),
  players: new Map(),

  addBuilding: (building: Building) => {
    set((state) => {
      const newBuildings = new Map(state.buildings);
      newBuildings.set(building.id, building);
      return { buildings: newBuildings };
    });
  },

  addResource: (resource: Resource) => {
    set((state) => {
      const newResources = new Map(state.resources);
      newResources.set(resource.id, resource);
      return { resources: newResources };
    });
  },

  detectResourceAtPosition: (position: Position): Resource | null => {
    const { resources } = get();
    
    for (const [_, resource] of resources) {
      const distance = Math.sqrt(
        Math.pow(resource.position.x - position.x, 2) +
        Math.pow(resource.position.y - position.y, 2)
      );
      
      if (distance <= RESOURCE_DETECTION_RADIUS) {
        return resource;
      }
    }
    
    return null;
  },

  setRallyPoint: (buildingId: string, position: Position) => {
    set((state) => {
      const newBuildings = new Map(state.buildings);
      const building = newBuildings.get(buildingId);
      
      if (!building) return state;

      const detectedResource = get().detectResourceAtPosition(position);
      
      const rallyPoint: RallyPoint = {
        position,
        targetResource: detectedResource || undefined,
        isResourceRallyPoint: detectedResource !== null,
      };

      building.rallyPoint = rallyPoint;
      newBuildings.set(buildingId, building);

      console.log(
        detectedResource
          ? `Rally point set on ${detectedResource.type} resource at (${position.x}, ${position.y})`
          : `Rally point set at (${position.x}, ${position.y})`
      );

      return { buildings: newBuildings };
    });
  },

  spawnUnit: (buildingId: string, unitType: string) => {
    const { buildings, units } = get();
    const building = buildings.get(buildingId);
    
    if (!building) return;

    const newUnit: Unit = {
      id: `unit_${Date.now()}_${Math.random()}`,
      type: unitType as 'worker' | 'soldier',
      position: { ...building.position },
      isGathering: false,
      ownerId: building.ownerId,
    };

    if (building.rallyPoint) {
      newUnit.position = { ...building.rallyPoint.position };

      if (building.rallyPoint.isResourceRallyPoint && building.rallyPoint.targetResource) {
        newUnit.isGathering = true;
        newUnit.targetResource = building.rallyPoint.targetResource.id;
        
        console.log(
          `Worker ${newUnit.id} spawned and assigned to gather from ${building.rallyPoint.targetResource.type}`
        );
      } else {
        console.log(`Unit ${newUnit.id} spawned at rally point (${newUnit.position.x}, ${newUnit.position.y})`);
      }
    } else {
      console.log(`Unit ${newUnit.id} spawned at building (${newUnit.position.x}, ${newUnit.position.y})`);
    }

    set((state) => {
      const newUnits = new Map(state.units);
      newUnits.set(newUnit.id, newUnit);
      return { units: newUnits };
    });
  },

  assignUnitToGather: (unitId: string, resourceId: string) => {
    set((state) => {
      const newUnits = new Map(state.units);
      const unit = newUnits.get(unitId);
      
      if (!unit) return state;

      unit.isGathering = true;
      unit.targetResource = resourceId;
      newUnits.set(unitId, unit);

      return { units: newUnits };
    });
  },
}));
