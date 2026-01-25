/**
 * Test for the releaseAllUnits bug fix
 * 
 * This test verifies that the "Release all" button correctly:
 * 1. Clears the building's garrisonedUnits array
 * 2. Updates each unit to remove their garrisonedIn property
 * 3. Sets exit positions for each unit
 */

import { useGameStore } from './gameStore';

describe('releaseAllUnits fix', () => {
  it('should correctly release all units from a building', () => {
    const store = useGameStore.getState();
    
    // Get initial state
    const buildingId = 'building-1';
    const initialBuilding = store.buildings[buildingId];
    const initialGarrisonedUnitIds = [...initialBuilding.garrisonedUnits];
    
    // Verify units are initially garrisoned
    initialGarrisonedUnitIds.forEach((unitId) => {
      const unit = store.units[unitId];
      expect(unit.garrisonedIn).toBe(buildingId);
    });
    
    // Release all units
    store.releaseAllUnits(buildingId);
    
    // Get updated state
    const updatedStore = useGameStore.getState();
    const updatedBuilding = updatedStore.buildings[buildingId];
    
    // Verify building's garrison list is empty
    expect(updatedBuilding.garrisonedUnits).toHaveLength(0);
    
    // Verify each unit has been properly released
    initialGarrisonedUnitIds.forEach((unitId) => {
      const unit = updatedStore.units[unitId];
      
      // Unit should no longer be garrisoned
      expect(unit.garrisonedIn).toBeUndefined();
      
      // Unit should have a new position (not the same as before)
      const oldUnit = store.units[unitId];
      expect(
        unit.position.x !== oldUnit.position.x || 
        unit.position.y !== oldUnit.position.y
      ).toBe(true);
    });
  });

  it('should handle empty garrison gracefully', () => {
    const store = useGameStore.getState();
    const buildingId = 'building-2'; // This building has no garrisoned units
    
    // This should not throw an error
    expect(() => {
      store.releaseAllUnits(buildingId);
    }).not.toThrow();
    
    const building = useGameStore.getState().buildings[buildingId];
    expect(building.garrisonedUnits).toHaveLength(0);
  });

  it('should position units in a circle around the building', () => {
    const store = useGameStore.getState();
    const buildingId = 'building-1';
    const building = store.buildings[buildingId];
    const unitIds = [...building.garrisonedUnits];
    
    // Release all units
    store.releaseAllUnits(buildingId);
    
    const updatedStore = useGameStore.getState();
    
    // Verify units are positioned around the building
    unitIds.forEach((unitId) => {
      const unit = updatedStore.units[unitId];
      const dx = unit.position.x - building.position.x;
      const dy = unit.position.y - building.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // Units should be approximately 60 pixels away (radius)
      expect(distance).toBeCloseTo(60, 0);
    });
  });
});
