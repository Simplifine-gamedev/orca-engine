# Orca RTS Game Components

This directory contains the game logic for the Orca RTS game built on the Orca Engine.

## Structure

```
src/
├── store/
│   ├── mobStore.ts          # Mob configurations and stats
│   └── mobStore.test.js     # Combat balance tests
├── COMBAT_BALANCE.md        # Detailed balance documentation
└── README.md                # This file
```

## Quick Start

### Mob Configuration

The mob store (`store/mobStore.ts`) contains all mob type configurations including:
- Health and armor values
- Attack damage and speed
- Movement speed
- Rewards (XP and gold)

Example usage:

```typescript
import { getMobConfig, calculateDamage, createMobInstance } from './store/mobStore';

// Get a mob configuration
const goblinConfig = getMobConfig('goblin');

// Create a new mob instance
const goblin = createMobInstance('goblin');

// Calculate damage with armor mitigation
const actualDamage = calculateDamage(50, goblin.armor, goblin.armorPercent);
```

### Server Integration

The game server (`../server/GameServer.js`) provides server-authoritative combat:

```javascript
const { getGameServer } = require('../server/GameServer');

const gameServer = getGameServer();

// Spawn a mob
const mobId = gameServer.spawnMob('goblin', { x: 100, y: 200 });

// Apply damage to mob
const result = gameServer.applyDamageToMob(mobId, 50, 'player1');

// Register players
gameServer.registerPlayer('player1', { position: { x: 0, y: 0 } });
```

## Available Mob Types

### Basic Mobs
- **Goblin**: Fast, lightly armored melee fighter (150 HP, 5+10% armor)
- **Orc Warrior**: Standard melee with moderate armor (250 HP, 10+15% armor)
- **Orc Archer**: Ranged attacker with low armor (120 HP, 3+5% armor)

### Elite Mobs
- **Orc Berserker**: High damage elite (400 HP, 15+20% armor)
- **Troll**: Very tanky with high armor (500 HP, 20+25% armor)

### Boss Mobs
- **Goblin Chief**: Boss-level goblin (800 HP, 25+30% armor)
- **Orc Warlord**: Major boss encounter (1200 HP, 35+35% armor)

## Testing

Run the combat balance tests:

```bash
node src/store/mobStore.test.js
```

This will verify:
- Damage calculation formulas
- Mob survivability against heavy soldiers
- Mob filtering and instance creation
- Combat scenarios

## Combat Balance

See [COMBAT_BALANCE.md](./COMBAT_BALANCE.md) for detailed information about:
- Balance philosophy and design decisions
- Armor system mechanics
- Combat examples and calculations
- Testing recommendations
- Future considerations

## Key Balance Changes (ORC-127)

The combat system was rebalanced to fix the issue where mobs died too quickly:

1. **Increased Health**: All mobs now have 3-5x more health
2. **Armor System**: Dual-layer armor (flat + percentage) for damage mitigation
3. **Survivability**: Mobs now survive 3-25 hits depending on type and difficulty

### Before vs After

| Mob Type | Old HP | New HP | Hits to Kill (Heavy Soldier) |
|----------|--------|--------|------------------------------|
| Goblin | ~40 | 150 | 1 hit → 4 hits |
| Orc Warrior | ~70 | 250 | 2 hits → 8 hits |

## Integration with Orca Engine

These TypeScript/JavaScript files are designed to integrate with the Orca Engine (Godot-based) through:
- WebSocket communication for real-time updates
- JSON serialization for state synchronization
- Event-driven architecture for combat resolution

## Development

### Adding New Mob Types

1. Add configuration to `MOB_CONFIGS` in `mobStore.ts`
2. Set appropriate health, armor, and damage values
3. Choose mob type: 'melee', 'ranged', 'elite', or 'boss'
4. Run tests to verify balance
5. Update documentation

### Modifying Balance

1. Adjust values in `mobStore.ts`
2. Run test suite: `node src/store/mobStore.test.js`
3. Verify hits-to-kill metrics
4. Update `COMBAT_BALANCE.md` if philosophy changes
5. Commit with clear balance notes

## Dependencies

- Node.js for server-side logic
- TypeScript for type safety (mob store)
- Orca Engine for rendering and game loop

## License

See LICENSE.txt in root directory
