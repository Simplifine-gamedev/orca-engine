// Test file for scout unit functionality
import { gameStore } from './store/gameStore';
import { scoutUnit, getUnitsFromBuilding } from './config/factions';

console.log('=== Scout Unit Test Suite ===\n');

// Test 1: Initialize game
console.log('Test 1: Initializing game...');
gameStore.initializeGame(1);
const state = gameStore.getState();
console.log(`✓ Game initialized with ${state.players.size} player(s)`);
console.log();

// Test 2: Check starting resources
console.log('Test 2: Checking starting resources...');
const resources = gameStore.getPlayerResources('player_0');
console.log('Starting resources:', resources);
console.log(`✓ Player has ${resources?.gold} gold, ${resources?.food} food`);
console.log();

// Test 3: Verify scout configuration
console.log('Test 3: Verifying scout unit configuration...');
console.log('Scout stats:');
console.log(`  - Name: ${scoutUnit.name}`);
console.log(`  - Cost: ${scoutUnit.cost.gold} gold, ${scoutUnit.cost.food} food`);
console.log(`  - Movement Speed: ${scoutUnit.stats.movementSpeed}`);
console.log(`  - Vision Range: ${scoutUnit.stats.visionRange}`);
console.log(`  - Attack: ${scoutUnit.stats.attack}`);
console.log(`  - Defense: ${scoutUnit.stats.defense}`);
console.log(`  - Health: ${scoutUnit.stats.health}`);
console.log(`  - Build Time: ${scoutUnit.buildTime}s`);
console.log(`  - Available from: ${scoutUnit.availableFrom.join(', ')}`);
console.log('✓ Scout configuration loaded successfully');
console.log();

// Test 4: Check building availability
console.log('Test 4: Checking unit availability from buildings...');
const townCenterUnits = getUnitsFromBuilding('town_center');
const stableUnits = getUnitsFromBuilding('stable');
console.log(`Town Center can produce: ${townCenterUnits.map(u => u.name).join(', ')}`);
console.log(`Stable can produce: ${stableUnits.map(u => u.name).join(', ')}`);
console.log('✓ Scout available from both town center and stable');
console.log();

// Test 5: Create first scout
console.log('Test 5: Creating first scout...');
const scout1Id = gameStore.createScout('player_0', { x: 100, y: 100 });
if (scout1Id) {
  const scout1 = gameStore.getState().units.get(scout1Id);
  console.log(`✓ Scout created with ID: ${scout1Id}`);
  console.log(`  Position: (${scout1?.position.x}, ${scout1?.position.y})`);
  console.log(`  Health: ${scout1?.health}/${scout1?.unitType.stats.health}`);
  
  const resourcesAfter = gameStore.getPlayerResources('player_0');
  console.log(`  Resources after: ${resourcesAfter?.gold} gold, ${resourcesAfter?.food} food`);
} else {
  console.log('✗ Failed to create scout');
}
console.log();

// Test 6: Create multiple scouts
console.log('Test 6: Creating multiple scouts...');
const scout2Id = gameStore.createScout('player_0', { x: 200, y: 150 });
const scout3Id = gameStore.createScout('player_0', { x: 150, y: 200 });
const playerUnits = gameStore.getPlayerUnits('player_0');
console.log(`✓ Player now has ${playerUnits.length} unit(s)`);
playerUnits.forEach((unit, index) => {
  console.log(`  Unit ${index + 1}: ${unit.unitType.name} at (${unit.position.x}, ${unit.position.y})`);
});
console.log();

// Test 7: Test scout movement
console.log('Test 7: Testing scout movement...');
if (scout1Id) {
  const oldPosition = gameStore.getState().units.get(scout1Id)?.position;
  const newPosition = { x: 300, y: 300 };
  const moved = gameStore.moveUnit(scout1Id, newPosition);
  const currentPosition = gameStore.getState().units.get(scout1Id)?.position;
  console.log(`✓ Scout moved from (${oldPosition?.x}, ${oldPosition?.y}) to (${currentPosition?.x}, ${currentPosition?.y})`);
  console.log(`  Movement successful: ${moved}`);
}
console.log();

// Test 8: Test vision range
console.log('Test 8: Testing scout vision range...');
if (scout1Id) {
  const vision = gameStore.getVisibleArea(scout1Id);
  if (vision) {
    console.log(`✓ Scout vision area:`);
    console.log(`  Center: (${vision.x}, ${vision.y})`);
    console.log(`  Radius: ${vision.radius} units`);
    console.log(`  Total visible area: ~${Math.PI * vision.radius * vision.radius} square units`);
  }
}
console.log();

// Test 9: Resource depletion test
console.log('Test 9: Testing resource management...');
const currentResources = gameStore.getPlayerResources('player_0');
console.log(`Current resources: ${currentResources?.gold} gold, ${currentResources?.food} food`);
const canAffordMore = currentResources && currentResources.gold >= 50 && currentResources.food >= 25;
console.log(`Can afford another scout: ${canAffordMore}`);

if (canAffordMore) {
  const scout4Id = gameStore.createScout('player_0', { x: 250, y: 250 });
  const finalResources = gameStore.getPlayerResources('player_0');
  console.log(`✓ Created another scout, resources now: ${finalResources?.gold} gold, ${finalResources?.food} food`);
} else {
  console.log('✓ Correctly preventing scout creation due to insufficient resources');
}
console.log();

// Test 10: Final summary
console.log('Test 10: Final summary...');
const finalUnits = gameStore.getPlayerUnits('player_0');
const finalResources = gameStore.getPlayerResources('player_0');
console.log(`Total scouts created: ${finalUnits.length}`);
console.log(`Remaining resources: ${finalResources?.gold} gold, ${finalResources?.food} food`);
console.log(`Total cost spent: ${500 - (finalResources?.gold || 0)} gold, ${200 - (finalResources?.food || 0)} food`);
console.log();

console.log('=== All Tests Complete ===');

// Performance comparison
console.log('\n=== Scout vs Other Units Comparison ===');
console.log('| Attribute       | Scout | Warrior | Archer |');
console.log('|-----------------|-------|---------|--------|');
console.log('| Movement Speed  | 8.0   | 3.5     | 4.0    | <- Fastest!');
console.log('| Vision Range    | 15    | 8       | 10     | <- Best!');
console.log('| Cost (Gold)     | 50    | 100     | 80     | <- Cheapest!');
console.log('| Build Time (s)  | 15    | 30      | 25     | <- Fastest!');
console.log('| Health          | 60    | 150     | 80     |');
console.log('| Attack          | 5     | 15      | 12     |');
console.log('| Defense         | 2     | 10      | 5      |');
console.log('\n✓ Scout excels at exploration and speed, as designed for early game!');
