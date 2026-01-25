// Example usage of scout units in Orca RTS
// This file demonstrates how to use the scout unit system

import { scoutUnitConfig, getUnitConfig, getUnitsFromBuilding } from '../config/factions';
import gameStore from '../store/gameStore';
import { getFactionLoader } from '../config/factionLoader';

console.log('=== Orca RTS Scout Unit Usage Example ===\n');

// Example 1: Basic Game Setup with Scouts
console.log('Example 1: Basic Game Setup');
console.log('---------------------------');
gameStore.reset();
gameStore.addPlayer('player1', 'Alice', 'base');
console.log('✓ Game initialized with player "Alice"');

// Example 2: Check Available Units from Town Center
console.log('\nExample 2: Available Units from Town Center');
console.log('--------------------------------------------');
const availableUnits = getUnitsFromBuilding('town_center');
console.log(`Available units from Town Center:`);
availableUnits.forEach(unit => {
  console.log(`  - ${unit.name}: ${JSON.stringify(unit.cost)}`);
});

// Example 3: Create Multiple Scout Units for Early Exploration
console.log('\nExample 3: Create Scout Units for Map Exploration');
console.log('--------------------------------------------------');
const scoutPositions = [
  { x: 100, y: 100, direction: 'north' },
  { x: 100, y: 100, direction: 'east' },
  { x: 100, y: 100, direction: 'south' },
  { x: 100, y: 100, direction: 'west' },
];

const createdScouts = [];
for (const pos of scoutPositions) {
  const scout = gameStore.createUnit('player1', 'scout', { x: pos.x, y: pos.y });
  if (scout) {
    createdScouts.push(scout);
    console.log(`✓ Created scout ${scout.id} heading ${pos.direction}`);
  }
}

// Example 4: Check Remaining Resources
console.log('\nExample 4: Resource Management');
console.log('------------------------------');
const player = gameStore.getState().players[0];
console.log(`Resources after creating ${createdScouts.length} scouts:`);
console.log(`  Gold: ${player.resources.gold}`);
console.log(`  Food: ${player.resources.food}`);
console.log(`  Wood: ${player.resources.wood}`);

// Example 5: Scout Group Selection and Movement
console.log('\nExample 5: Scout Group Control');
console.log('------------------------------');
const scoutIds = createdScouts.map(s => s.id);
gameStore.selectUnits(scoutIds);
console.log(`✓ Selected ${scoutIds.length} scouts`);

// Move scouts to exploration positions
const explorationTargets = [
  { x: 200, y: 50 },
  { x: 200, y: 100 },
  { x: 200, y: 150 },
  { x: 200, y: 200 },
];

createdScouts.forEach((scout, index) => {
  const target = explorationTargets[index];
  gameStore.moveUnit(scout.id, target);
  console.log(`✓ Scout ${index + 1} moving to (${target.x}, ${target.y})`);
});

// Example 6: Load Faction-Specific Scouts
console.log('\nExample 6: Faction-Specific Scout Units');
console.log('----------------------------------------');
const loader = getFactionLoader('../generated_factions');
const allScouts = loader.getScoutUnits();

console.log(`Loaded ${allScouts.length} faction-specific scout variants:\n`);
allScouts.forEach(scout => {
  console.log(`${scout.name} (${scout.faction}):`);
  console.log(`  Speed: ${scout.stats.movementSpeed} | Vision: ${scout.stats.visionRange}`);
  console.log(`  Cost: ${JSON.stringify(scout.cost)}`);
  if (scout.specialAbilities && scout.specialAbilities.length > 0) {
    console.log(`  Special: ${scout.specialAbilities[0].name}`);
  }
  console.log();
});

// Example 7: Scout Statistics Summary
console.log('Example 7: Scout Statistics Summary');
console.log('-----------------------------------');
const stats = loader.getStatsSummary();
console.log(`Total Units: ${stats.totalUnits}`);
console.log(`Scout Units: ${stats.scoutUnits}`);
console.log(`Average Scout Speed: ${stats.averageScoutSpeed.toFixed(2)}`);
console.log(`Average Scout Vision: ${stats.averageScoutVision.toFixed(2)}`);
console.log(`Average Scout Cost: ${Math.round(stats.averageScoutCost.gold + stats.averageScoutCost.food)} resources`);

// Example 8: Early Game Scout Rush Strategy
console.log('\nExample 8: Early Game Scout Rush Strategy');
console.log('-----------------------------------------');
gameStore.reset();
gameStore.addPlayer('player2', 'Bob', 'base');
gameStore.addResources('player2', { gold: 1000, food: 1000 }); // Give extra resources

// Build 5 scouts for aggressive early exploration
console.log('Building 5 scouts for early map control...');
const scoutRush = [];
for (let i = 0; i < 5; i++) {
  const scout = gameStore.createUnit('player2', 'scout', { 
    x: 100 + (i * 10), 
    y: 100 
  });
  if (scout) {
    scoutRush.push(scout);
  }
}
console.log(`✓ Created ${scoutRush.length} scouts`);
console.log('Strategy: Use scouts to:');
console.log('  1. Reveal entire map quickly');
console.log('  2. Find enemy base location');
console.log('  3. Identify resource locations');
console.log('  4. Spot enemy unit movements');

// Example 9: Vision Range Comparison
console.log('\nExample 9: Vision Range Comparison');
console.log('----------------------------------');
const scoutVision = scoutUnitConfig.stats.visionRange;
console.log(`Scout vision range: ${scoutVision} units`);
console.log(`Typical warrior vision: ~6 units (${Math.round((scoutVision / 6) * 100)}% smaller)`);
console.log(`Scout explores ~${Math.round(Math.PI * scoutVision * scoutVision / 100)} times more area`);

// Example 10: Cost-Effectiveness Analysis
console.log('\nExample 10: Cost-Effectiveness Analysis');
console.log('---------------------------------------');
const scoutCost = (scoutUnitConfig.cost.gold || 0) + (scoutUnitConfig.cost.food || 0);
const typicalWarriorCost = 150; // Example warrior cost
console.log(`Scout cost: ${scoutCost} resources`);
console.log(`Typical warrior cost: ${typicalWarriorCost} resources`);
console.log(`Can build ${Math.floor(typicalWarriorCost / scoutCost)} scouts for 1 warrior's cost`);
console.log(`Vision coverage: ${Math.floor(typicalWarriorCost / scoutCost) * scoutVision} units vs ${6} units`);

console.log('\n=== Examples Complete ===');
console.log('\nKey Takeaways:');
console.log('1. Scouts are cheap and fast - perfect for early exploration');
console.log('2. Large vision range reveals more map per unit');
console.log('3. Multiple scouts can cover map quickly');
console.log('4. Each faction has unique scout abilities');
console.log('5. Scouts are cost-effective for map control');
