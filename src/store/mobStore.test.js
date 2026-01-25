/**
 * Combat Balance Tests
 * Tests to verify mob balance changes for ORC-127
 */

const {
  MOB_CONFIGS,
  calculateDamage,
  createMobInstance,
  getMobConfig,
  getAllMobs,
  getMobsByType
} = require('./mobStore.ts');

// Test configurations
const HEAVY_SOLDIER_DAMAGE = 50;
const LIGHT_SOLDIER_DAMAGE = 25;
const ARCHER_DAMAGE = 30;

/**
 * Test that mobs don't get one-shot by heavy soldiers
 */
function testMobSurvivability() {
  console.log('\n=== Testing Mob Survivability ===\n');
  
  const mobsToTest = ['goblin', 'orc', 'orc_archer', 'orc_berserker', 'troll'];
  let allPassed = true;

  mobsToTest.forEach(mobType => {
    const config = getMobConfig(mobType);
    if (!config) {
      console.error(`❌ Mob config not found: ${mobType}`);
      allPassed = false;
      return;
    }

    const actualDamage = calculateDamage(
      HEAVY_SOLDIER_DAMAGE,
      config.stats.armor,
      config.stats.armorPercent
    );

    const hitsToKill = Math.ceil(config.stats.health / actualDamage);
    const isBalanced = hitsToKill >= 3; // Minimum 3 hits to kill

    const status = isBalanced ? '✓' : '✗';
    console.log(`${status} ${config.name}`);
    console.log(`  Health: ${config.stats.health}`);
    console.log(`  Armor: ${config.stats.armor} + ${config.stats.armorPercent * 100}%`);
    console.log(`  Damage taken: ${actualDamage} (reduced from ${HEAVY_SOLDIER_DAMAGE})`);
    console.log(`  Hits to kill: ${hitsToKill}`);
    console.log(`  ${isBalanced ? 'PASS' : 'FAIL'} - ${isBalanced ? 'Survives multiple hits' : 'Dies too quickly'}\n`);

    if (!isBalanced) allPassed = false;
  });

  return allPassed;
}

/**
 * Test damage calculation formula
 */
function testDamageCalculation() {
  console.log('\n=== Testing Damage Calculation ===\n');
  
  const tests = [
    {
      name: 'No armor',
      baseDamage: 50,
      armor: 0,
      armorPercent: 0,
      expected: 50
    },
    {
      name: 'Flat armor only',
      baseDamage: 50,
      armor: 10,
      armorPercent: 0,
      expected: 40
    },
    {
      name: 'Percentage armor only',
      baseDamage: 50,
      armor: 0,
      armorPercent: 0.2,
      expected: 40
    },
    {
      name: 'Combined armor',
      baseDamage: 50,
      armor: 10,
      armorPercent: 0.2,
      expected: 32 // (50-10) * 0.8 = 32
    },
    {
      name: 'High armor (goblin)',
      baseDamage: 50,
      armor: 5,
      armorPercent: 0.1,
      expected: 40 // (50-5) * 0.9 = 40.5 -> 40
    },
    {
      name: 'Minimum damage',
      baseDamage: 10,
      armor: 100,
      armorPercent: 0.9,
      expected: 1 // Should never be less than 1
    }
  ];

  let allPassed = true;

  tests.forEach(test => {
    const actual = calculateDamage(test.baseDamage, test.armor, test.armorPercent);
    const passed = actual === test.expected;
    
    console.log(`${passed ? '✓' : '✗'} ${test.name}`);
    console.log(`  Base: ${test.baseDamage}, Armor: ${test.armor} + ${test.armorPercent * 100}%`);
    console.log(`  Expected: ${test.expected}, Actual: ${actual}`);
    console.log(`  ${passed ? 'PASS' : 'FAIL'}\n`);

    if (!passed) allPassed = false;
  });

  return allPassed;
}

/**
 * Test combat scenarios
 */
function testCombatScenarios() {
  console.log('\n=== Testing Combat Scenarios ===\n');

  const scenarios = [
    {
      name: '1 Heavy Soldier vs Goblin',
      mobType: 'goblin',
      unitDamage: HEAVY_SOLDIER_DAMAGE,
      unitCount: 1,
      expectedResult: 'Should require 4+ attacks'
    },
    {
      name: '1 Heavy Soldier vs Orc Warrior',
      mobType: 'orc',
      unitDamage: HEAVY_SOLDIER_DAMAGE,
      unitCount: 1,
      expectedResult: 'Should require 7+ attacks'
    },
    {
      name: '2 Light Soldiers vs Goblin',
      mobType: 'goblin',
      unitDamage: LIGHT_SOLDIER_DAMAGE,
      unitCount: 2,
      expectedResult: 'Should require 4+ attacks total'
    },
    {
      name: '1 Archer vs Orc Berserker',
      mobType: 'orc_berserker',
      unitDamage: ARCHER_DAMAGE,
      unitCount: 1,
      expectedResult: 'Should require 20+ attacks'
    }
  ];

  scenarios.forEach(scenario => {
    const config = getMobConfig(scenario.mobType);
    if (!config) {
      console.error(`❌ Mob not found: ${scenario.mobType}`);
      return;
    }

    const actualDamage = calculateDamage(
      scenario.unitDamage,
      config.stats.armor,
      config.stats.armorPercent
    );

    const damagePerRound = actualDamage * scenario.unitCount;
    const roundsToKill = Math.ceil(config.stats.health / damagePerRound);
    const attacksToKill = roundsToKill * scenario.unitCount;

    console.log(`📊 ${scenario.name}`);
    console.log(`  Mob Health: ${config.stats.health}`);
    console.log(`  Damage per unit: ${actualDamage} (from ${scenario.unitDamage})`);
    console.log(`  Total damage per round: ${damagePerRound}`);
    console.log(`  Rounds to kill: ${roundsToKill}`);
    console.log(`  Total attacks needed: ${attacksToKill}`);
    console.log(`  Expected: ${scenario.expectedResult}\n`);
  });
}

/**
 * Test mob type filtering
 */
function testMobFiltering() {
  console.log('\n=== Testing Mob Type Filtering ===\n');
  
  const types = ['melee', 'ranged', 'elite', 'boss'];
  let allPassed = true;

  types.forEach(type => {
    const mobs = getMobsByType(type);
    console.log(`${type.toUpperCase()} mobs: ${mobs.length}`);
    mobs.forEach(mob => {
      console.log(`  - ${mob.name} (HP: ${mob.stats.health})`);
    });
    console.log();
  });

  const allMobs = getAllMobs();
  console.log(`Total mobs configured: ${allMobs.length}\n`);

  return allPassed;
}

/**
 * Test mob instance creation
 */
function testMobInstanceCreation() {
  console.log('\n=== Testing Mob Instance Creation ===\n');
  
  const mobTypes = ['goblin', 'orc', 'troll'];
  let allPassed = true;

  mobTypes.forEach(mobType => {
    const instance = createMobInstance(mobType);
    const config = getMobConfig(mobType);
    
    if (!instance || !config) {
      console.error(`❌ Failed to create instance for ${mobType}`);
      allPassed = false;
      return;
    }

    const healthMatches = instance.health === instance.maxHealth;
    const statsMatch = instance.health === config.stats.maxHealth;

    console.log(`${healthMatches && statsMatch ? '✓' : '✗'} ${config.name}`);
    console.log(`  Health: ${instance.health}/${instance.maxHealth}`);
    console.log(`  Armor: ${instance.armor} + ${instance.armorPercent * 100}%`);
    console.log(`  Damage: ${instance.damage}`);
    console.log(`  ${healthMatches && statsMatch ? 'PASS' : 'FAIL'}\n`);

    if (!healthMatches || !statsMatch) allPassed = false;
  });

  return allPassed;
}

/**
 * Generate balance report
 */
function generateBalanceReport() {
  console.log('\n=== Combat Balance Report ===\n');
  console.log('Heavy Soldier Damage: ' + HEAVY_SOLDIER_DAMAGE);
  console.log('Light Soldier Damage: ' + LIGHT_SOLDIER_DAMAGE);
  console.log('Archer Damage: ' + ARCHER_DAMAGE);
  console.log('\nMob Statistics:\n');

  const allMobs = getAllMobs();
  allMobs.forEach(mob => {
    const heavyDamage = calculateDamage(
      HEAVY_SOLDIER_DAMAGE,
      mob.stats.armor,
      mob.stats.armorPercent
    );
    const hitsToKill = Math.ceil(mob.stats.health / heavyDamage);

    console.log(`${mob.name} (${mob.type})`);
    console.log(`  Health: ${mob.stats.health}`);
    console.log(`  Armor: ${mob.stats.armor} flat + ${mob.stats.armorPercent * 100}%`);
    console.log(`  Damage taken from heavy soldier: ${heavyDamage}`);
    console.log(`  Hits to kill (heavy soldier): ${hitsToKill}`);
    console.log(`  Damage output: ${mob.stats.damage}`);
    console.log(`  Attack speed: ${mob.stats.attackSpeed}/s`);
    console.log(`  Move speed: ${mob.stats.moveSpeed}`);
    console.log(`  XP/Gold: ${mob.stats.xpReward}/${mob.stats.goldReward}`);
    console.log();
  });
}

/**
 * Run all tests
 */
function runAllTests() {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║  Combat Balance Test Suite - ORC-127          ║');
  console.log('╚════════════════════════════════════════════════╝');

  const results = {
    damageCalculation: testDamageCalculation(),
    mobSurvivability: testMobSurvivability(),
    mobFiltering: testMobFiltering(),
    instanceCreation: testMobInstanceCreation()
  };

  // Run scenario tests (informational, no pass/fail)
  testCombatScenarios();
  
  // Generate report
  generateBalanceReport();

  // Summary
  console.log('\n╔════════════════════════════════════════════════╗');
  console.log('║  Test Summary                                  ║');
  console.log('╚════════════════════════════════════════════════╝\n');

  Object.entries(results).forEach(([test, passed]) => {
    console.log(`${passed ? '✓' : '✗'} ${test}: ${passed ? 'PASSED' : 'FAILED'}`);
  });

  const allPassed = Object.values(results).every(r => r);
  console.log(`\n${allPassed ? '✓' : '✗'} Overall: ${allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED'}\n`);

  return allPassed;
}

// Run tests if executed directly
if (require.main === module) {
  const success = runAllTests();
  process.exit(success ? 0 : 1);
}

module.exports = {
  testDamageCalculation,
  testMobSurvivability,
  testCombatScenarios,
  testMobFiltering,
  testMobInstanceCreation,
  runAllTests
};
