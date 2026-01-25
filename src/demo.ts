/**
 * Demo script showing rally point to resource feature
 */

import { gameStore } from './store/gameStore';

console.log('=== Rally Point to Resource Demo ===\n');

// 1. Create a gold mine resource
console.log('1. Creating gold mine at (300, 200)...');
gameStore.addResource({
  id: 'gold-mine-1',
  type: 'gold',
  position: { x: 300, y: 200 },
  amount: 5000,
});

// 2. Create a town hall building
console.log('2. Creating town hall at (100, 100)...');
gameStore.addBuilding({
  id: 'townhall-1',
  type: 'townhall',
  position: { x: 100, y: 100 },
  playerId: 'player-1',
  spawnQueue: [],
});

// 3. Set rally point on the gold mine
console.log('3. Setting rally point on gold mine...');
gameStore.setRallyPoint('townhall-1', { x: 300, y: 200 });

const building = gameStore.getState().buildings.get('townhall-1');
if (building?.rallyPoint?.targetResourceId) {
  console.log('✓ Rally point successfully set on resource!');
  console.log(`  Resource Type: ${building.rallyPoint.targetResource?.type}`);
  console.log(`  Resource ID: ${building.rallyPoint.targetResourceId}`);
} else {
  console.log('✗ Rally point not on resource');
}

// 4. Spawn a worker
console.log('\n4. Spawning worker from town hall...');
const worker = gameStore.spawnUnit('townhall-1', 'worker');

if (worker) {
  console.log('✓ Worker spawned successfully!');
  console.log(`  Worker ID: ${worker.id}`);
  console.log(`  Is Gathering: ${worker.isGathering}`);
  console.log(`  Target Resource: ${worker.targetResourceId}`);
  console.log(`  Position: (${worker.position.x}, ${worker.position.y})`);
  
  if (worker.isGathering && worker.targetResourceId === 'gold-mine-1') {
    console.log('\n✓✓✓ SUCCESS! Worker automatically assigned to gather from gold mine!');
  }
} else {
  console.log('✗ Failed to spawn worker');
}

// 5. Test rally point away from resources
console.log('\n5. Testing rally point away from resources...');
gameStore.setRallyPoint('townhall-1', { x: 500, y: 500 });

const building2 = gameStore.getState().buildings.get('townhall-1');
if (building2?.rallyPoint && !building2.rallyPoint.targetResourceId) {
  console.log('✓ Rally point set on empty location (no resource detected)');
} else {
  console.log('✗ Unexpected rally point state');
}

// 6. Spawn worker at non-resource rally point
console.log('6. Spawning worker at non-resource rally point...');
const worker2 = gameStore.spawnUnit('townhall-1', 'worker');

if (worker2 && !worker2.isGathering) {
  console.log('✓ Worker spawned at rally point without auto-gathering');
  console.log(`  Is Gathering: ${worker2.isGathering}`);
  console.log(`  Position: (${worker2.position.x}, ${worker2.position.y})`);
}

// Summary
console.log('\n=== Demo Complete ===');
console.log('\nFeatures Demonstrated:');
console.log('✓ Resource creation');
console.log('✓ Building creation');
console.log('✓ Rally point on resource detection');
console.log('✓ Automatic worker assignment to resource');
console.log('✓ Rally point on empty location');
console.log('✓ Normal worker spawn without auto-gathering');
