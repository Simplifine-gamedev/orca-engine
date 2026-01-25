# Combat Balance Documentation

## Overview
This document explains the combat balance changes implemented to fix the issue where mobs were dying too quickly (ORC-127).

## Problem
- Heavy soldiers were one-shotting or nearly one-shotting goblins and orcs
- Combat encounters ended too quickly
- Mobs provided no challenge to players

## Solution

### 1. Increased Health Values
All mob health values have been significantly increased:

| Mob Type | Old Health (estimated) | New Health | Survivability vs Heavy Soldier (50 dmg) |
|----------|----------------------|------------|----------------------------------------|
| Goblin | 30-50 | 150 | 3-4 hits |
| Orc Warrior | 60-80 | 250 | 5-6 hits |
| Orc Archer | 40-60 | 120 | 3 hits |
| Orc Berserker | - | 400 | 7-8 hits |
| Troll | - | 500 | 9-10 hits |
| Goblin Chief | - | 800 | 14-15 hits |
| Orc Warlord | - | 1200 | 20+ hits |

### 2. Armor System
Implemented a dual-layer armor system:

#### Flat Armor Reduction
- Reduces damage by a flat amount before percentage calculation
- Example: 10 armor reduces 50 damage to 40 damage

#### Percentage Armor Reduction
- Reduces remaining damage by a percentage
- Example: 15% armor reduces 40 damage to 34 damage

#### Combined Formula
```
actualDamage = max(1, floor((baseDamage - flatArmor) * (1 - armorPercent)))
```

### 3. Mob Armor Values

| Mob Type | Flat Armor | Armor % | Effective Damage Reduction (50 dmg base) |
|----------|-----------|---------|----------------------------------------|
| Goblin | 5 | 10% | 40.5 damage (19% reduction) |
| Orc Warrior | 10 | 15% | 34 damage (32% reduction) |
| Orc Archer | 3 | 5% | 44.65 damage (11% reduction) |
| Orc Berserker | 15 | 20% | 28 damage (44% reduction) |
| Troll | 20 | 25% | 22.5 damage (55% reduction) |
| Goblin Chief | 25 | 30% | 17.5 damage (65% reduction) |
| Orc Warlord | 35 | 35% | 9.75 damage (80% reduction) |

## Combat Example: Heavy Soldier vs Goblin

### Before Fix
- Heavy Soldier damage: 50
- Goblin health: 30-50
- Result: **One-shot kill**

### After Fix
- Heavy Soldier damage: 50
- Goblin armor: 5 flat, 10%
- Actual damage: (50 - 5) * 0.9 = 40.5
- Goblin health: 150
- **Hits to kill: 4 hits** (150 / 40.5 ≈ 3.7)

## Combat Example: Heavy Soldier vs Orc Warrior

### Before Fix
- Heavy Soldier damage: 50
- Orc health: 60-80
- Result: **1-2 hit kill**

### After Fix
- Heavy Soldier damage: 50
- Orc armor: 10 flat, 15%
- Actual damage: (50 - 10) * 0.85 = 34
- Orc health: 250
- **Hits to kill: 8 hits** (250 / 34 ≈ 7.4)

## Balance Philosophy

### Basic Mobs (Goblin, Orc Archer)
- Should survive 3-4 hits from heavy soldiers
- Provide quick but meaningful encounters
- Good for early game and farming

### Standard Mobs (Orc Warrior)
- Should survive 6-8 hits
- Require tactical consideration
- Good for mid-game content

### Elite Mobs (Orc Berserker, Troll)
- Should survive 8-12 hits
- Require focus fire or multiple units
- Challenge for experienced players

### Boss Mobs (Goblin Chief, Orc Warlord)
- Should survive 15-25 hits
- Require coordinated attacks
- End-game content

## Future Considerations

### Damage Types
Consider implementing:
- Physical damage (affected by armor)
- Magic damage (bypasses armor)
- Piercing damage (ignores % of armor)

### Mob Abilities
- Healing/regeneration for trolls
- Shield abilities for elite units
- Enrage mechanics for low health

### Player Unit Variety
- Light units: Fast attack, low damage
- Heavy units: Slow attack, high damage
- Ranged units: High damage, positioning important
- Magic units: Bypass armor

## Testing Recommendations

1. Test goblin encounters with 1-2 heavy soldiers
2. Test orc encounters with 3-4 heavy soldiers
3. Test elite mob encounters with full squad (6-8 units)
4. Test boss encounters with multiple squads
5. Monitor average combat duration
6. Collect player feedback on difficulty

## Configuration Files

- **Mob Stats**: `/src/store/mobStore.ts`
- **Server Logic**: `/server/GameServer.js`
- **Tests**: `/src/store/__tests__/mobStore.test.ts` (to be created)

## Rollback Plan

If balance proves too difficult:
1. Reduce health values by 20-30%
2. Reduce armor percentages by 5%
3. Keep flat armor values the same

## Monitoring Metrics

Track these metrics to evaluate balance:
- Average combat duration
- Player unit losses per encounter
- Mob kill/death ratio
- Player frustration reports
- Combat engagement metrics
