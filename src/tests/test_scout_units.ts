// Test file for scout units functionality
// Run with: npm test

import { scoutUnitConfig, getUnitConfig, getUnitsFromBuilding } from '../config/factions';
import gameStore from '../store/gameStore';

console.log('=== Orca RTS Scout Unit Test ===\n');

// Test 1: Verify scout unit configuration
console.log('Test 1: Scout Unit Configuration');
console.log('--------------------------------');
console.log(`Name: ${scoutUnitConfig.name}`);
console.log(`Type: ${scoutUnitConfig.type}`);
console.log(`Cost: ${JSON.stringify(scoutUnitConfig.cost)}`);
console.log(`Movement Speed: ${scoutUnitConfig.stats.movementSpeed}`);
console.log(`Vision Range: ${scoutUnitConfig.stats.visionRange}`);
console.log(`Attack: ${scoutUnitConfig.stats.attack}`);
console.log(`Build Time: ${scoutUnitConfig.buildTime}s`);
console.log(`Available From: ${scoutUnitConfig.availableFrom.join(', ')}`);
console.log('✓ Scout configuration loaded successfully\n');

// Test 2: Get unit by id
console.log('Test 2: Get Unit by ID');
console.log('----------------------');
const scout = getUnitConfig('scout');
if (scout) {
  console.log(`✓ Found unit: ${scout.name}`);
} else {
  console.error('✗ Failed to find scout unit');
}
console.log();

// Test 3: Get units from building
console.log('Test 3: Units from Town Center');
console.log('-------------------------------');
const townCenterUnits = getUnitsFromBuilding('town_center');
console.log(`Units available: ${townCenterUnits.map(u => u.name).join(', ')}`);
console.log(`✓ Found ${townCenterUnits.length} unit(s) from town center\n`);

// Test 4: Game store - create player and unit
console.log('Test 4: Game Store - Create Player and Scout');
console.log('---------------------------------------------');
gameStore.reset();
gameStore.addPlayer('player1', 'Test Player', 'base');
console.log('✓ Player created');

const state = gameStore.getState();
const player = state.players[0];
console.log(`Player resources: Gold=${player.resources.gold}, Food=${player.resources.food}`);

// Test 5: Create scout unit
console.log('\nTest 5: Create Scout Unit');
console.log('-------------------------');
const scoutUnit = gameStore.createUnit('player1', 'scout', { x: 100, y: 100 });
if (scoutUnit) {
  console.log(`✓ Scout created with ID: ${scoutUnit.id}`);
  console.log(`  Position: (${scoutUnit.position.x}, ${scoutUnit.position.y})`);
  console.log(`  Health: ${scoutUnit.health}/${scoutUnit.maxHealth}`);
  console.log(`  State: ${scoutUnit.state}`);
  
  const updatedPlayer = gameStore.getState().players[0];
  console.log(`Updated resources: Gold=${updatedPlayer.resources.gold}, Food=${updatedPlayer.resources.food}`);
  console.log(`Cost deducted: Gold=${player.resources.gold - updatedPlayer.resources.gold}, Food=${player.resources.food - updatedPlayer.resources.food}`);
} else {
  console.error('✗ Failed to create scout unit');
}
console.log();

// Test 6: Unit selection
console.log('Test 6: Unit Selection');
console.log('----------------------');
if (scoutUnit) {
  gameStore.selectUnits([scoutUnit.id]);
  const selectedUnits = gameStore.getState().selectedUnits;
  console.log(`✓ Selected ${selectedUnits.length} unit(s)`);
  const unit = gameStore.getUnit(scoutUnit.id);
  console.log(`  Unit selected state: ${unit?.isSelected}`);
}
console.log();

// Test 7: Scout characteristics validation
console.log('Test 7: Scout Characteristics Validation');
console.log('-----------------------------------------');
const validations = [
  { check: 'Fast movement', pass: scoutUnitConfig.stats.movementSpeed >= 8 },
  { check: 'Low cost', pass: (scoutUnitConfig.cost.gold || 0) + (scoutUnitConfig.cost.food || 0) <= 100 },
  { check: 'Large vision range', pass: scoutUnitConfig.stats.visionRange >= 10 },
  { check: 'Low attack', pass: scoutUnitConfig.stats.attack <= 10 },
  { check: 'Available from town center', pass: scoutUnitConfig.availableFrom.includes('town_center') },
];

validations.forEach(v => {
  console.log(`${v.pass ? '✓' : '✗'} ${v.check}`);
});

const allPassed = validations.every(v => v.pass);
console.log(`\n${allPassed ? '✓' : '✗'} All validation checks ${allPassed ? 'passed' : 'failed'}`);

// Test 8: Try to create unit without resources
console.log('\nTest 8: Insufficient Resources Test');
console.log('------------------------------------');
// Drain resources
gameStore.deductResources('player1', { gold: 1000, food: 1000 });
const poorPlayer = gameStore.getState().players[0];
console.log(`Player resources: Gold=${poorPlayer.resources.gold}, Food=${poorPlayer.resources.food}`);

const failedUnit = gameStore.createUnit('player1', 'scout', { x: 200, y: 200 });
if (!failedUnit) {
  console.log('✓ Correctly prevented unit creation with insufficient resources');
} else {
  console.error('✗ Should have failed to create unit');
}

console.log('\n=== All Tests Complete ===');
