import React, { useEffect, useState } from 'react';

export interface MobLairConfig {
  id: string;
  type: LairType;
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  spawnInterval: number; // milliseconds
  maxMobs: number;
  spawnRadius: number;
  lootTable?: LootItem[];
  destroyed: boolean;
}

export enum LairType {
  GOBLIN_CAMP = 'goblin_camp',
  OGRE_CAVE = 'ogre_cave',
  WOLF_DEN = 'wolf_den',
  BANDIT_HIDEOUT = 'bandit_hideout',
  UNDEAD_CRYPT = 'undead_crypt',
}

export interface LootItem {
  itemId: string;
  chance: number; // 0-1
  quantity: { min: number; max: number };
}

export interface MobSpawnConfig {
  mobType: string;
  level: { min: number; max: number };
  count: number;
}

// Lair type configurations
export const LAIR_CONFIGS: Record<LairType, {
  displayName: string;
  maxHealth: number;
  spawnInterval: number;
  maxMobs: number;
  spawnRadius: number;
  mobSpawns: MobSpawnConfig[];
  lootTable: LootItem[];
  appearance: {
    color: string;
    size: number;
  };
}> = {
  [LairType.GOBLIN_CAMP]: {
    displayName: 'Goblin Camp',
    maxHealth: 500,
    spawnInterval: 30000, // 30 seconds
    maxMobs: 5,
    spawnRadius: 100,
    mobSpawns: [
      { mobType: 'goblin_warrior', level: { min: 1, max: 3 }, count: 2 },
      { mobType: 'goblin_archer', level: { min: 1, max: 2 }, count: 1 },
    ],
    lootTable: [
      { itemId: 'gold', chance: 1.0, quantity: { min: 50, max: 100 } },
      { itemId: 'goblin_dagger', chance: 0.5, quantity: { min: 1, max: 1 } },
      { itemId: 'crude_armor', chance: 0.3, quantity: { min: 1, max: 1 } },
    ],
    appearance: {
      color: '#8B4513',
      size: 80,
    },
  },
  [LairType.OGRE_CAVE]: {
    displayName: 'Ogre Cave',
    maxHealth: 1200,
    spawnInterval: 60000, // 60 seconds
    maxMobs: 3,
    spawnRadius: 150,
    mobSpawns: [
      { mobType: 'ogre', level: { min: 5, max: 8 }, count: 1 },
      { mobType: 'cave_troll', level: { min: 4, max: 6 }, count: 1 },
    ],
    lootTable: [
      { itemId: 'gold', chance: 1.0, quantity: { min: 200, max: 500 } },
      { itemId: 'ogre_club', chance: 0.4, quantity: { min: 1, max: 1 } },
      { itemId: 'heavy_armor', chance: 0.2, quantity: { min: 1, max: 1 } },
      { itemId: 'rare_gem', chance: 0.1, quantity: { min: 1, max: 3 } },
    ],
    appearance: {
      color: '#2F4F4F',
      size: 120,
    },
  },
  [LairType.WOLF_DEN]: {
    displayName: 'Wolf Den',
    maxHealth: 400,
    spawnInterval: 20000, // 20 seconds
    maxMobs: 8,
    spawnRadius: 120,
    mobSpawns: [
      { mobType: 'wolf', level: { min: 2, max: 4 }, count: 3 },
      { mobType: 'dire_wolf', level: { min: 4, max: 6 }, count: 1 },
    ],
    lootTable: [
      { itemId: 'wolf_pelt', chance: 0.8, quantity: { min: 1, max: 3 } },
      { itemId: 'wolf_fang', chance: 0.6, quantity: { min: 1, max: 2 } },
      { itemId: 'leather_scraps', chance: 1.0, quantity: { min: 3, max: 8 } },
    ],
    appearance: {
      color: '#696969',
      size: 60,
    },
  },
  [LairType.BANDIT_HIDEOUT]: {
    displayName: 'Bandit Hideout',
    maxHealth: 600,
    spawnInterval: 40000, // 40 seconds
    maxMobs: 6,
    spawnRadius: 100,
    mobSpawns: [
      { mobType: 'bandit', level: { min: 3, max: 5 }, count: 2 },
      { mobType: 'bandit_rogue', level: { min: 4, max: 6 }, count: 1 },
      { itemId: 'bandit_chief', level: { min: 6, max: 8 }, count: 1 },
    ],
    lootTable: [
      { itemId: 'gold', chance: 1.0, quantity: { min: 100, max: 300 } },
      { itemId: 'stolen_goods', chance: 0.7, quantity: { min: 1, max: 5 } },
      { itemId: 'bandit_sword', chance: 0.4, quantity: { min: 1, max: 1 } },
      { itemId: 'lockpick_set', chance: 0.3, quantity: { min: 1, max: 1 } },
    ],
    appearance: {
      color: '#8B0000',
      size: 90,
    },
  },
  [LairType.UNDEAD_CRYPT]: {
    displayName: 'Undead Crypt',
    maxHealth: 800,
    spawnInterval: 45000, // 45 seconds
    maxMobs: 10,
    spawnRadius: 140,
    mobSpawns: [
      { mobType: 'skeleton', level: { min: 3, max: 5 }, count: 3 },
      { mobType: 'zombie', level: { min: 2, max: 4 }, count: 2 },
      { mobType: 'ghoul', level: { min: 5, max: 7 }, count: 1 },
    ],
    lootTable: [
      { itemId: 'bone_fragments', chance: 1.0, quantity: { min: 5, max: 15 } },
      { itemId: 'cursed_amulet', chance: 0.3, quantity: { min: 1, max: 1 } },
      { itemId: 'ancient_scroll', chance: 0.2, quantity: { min: 1, max: 2 } },
      { itemId: 'soul_gem', chance: 0.1, quantity: { min: 1, max: 1 } },
    ],
    appearance: {
      color: '#4B0082',
      size: 100,
    },
  },
};

interface MobLairProps {
  lair: MobLairConfig;
  onDestroy?: (lairId: string) => void;
  onTakeDamage?: (lairId: string, damage: number) => void;
  onClick?: (lairId: string) => void;
}

export const MobLair: React.FC<MobLairProps> = ({
  lair,
  onDestroy,
  onTakeDamage,
  onClick,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const config = LAIR_CONFIGS[lair.type];

  useEffect(() => {
    if (lair.destroyed && onDestroy) {
      onDestroy(lair.id);
    }
  }, [lair.destroyed, lair.id, onDestroy]);

  const handleClick = () => {
    if (onClick && !lair.destroyed) {
      onClick(lair.id);
    }
  };

  const healthPercentage = (lair.health / lair.maxHealth) * 100;

  if (lair.destroyed) {
    return null;
  }

  return (
    <div
      className="mob-lair"
      style={{
        position: 'absolute',
        left: lair.position.x,
        top: lair.position.y,
        width: config.appearance.size,
        height: config.appearance.size,
        cursor: 'pointer',
        transform: 'translate(-50%, -50%)',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
    >
      {/* Spawn radius indicator (visible on hover) */}
      {isHovered && (
        <div
          className="spawn-radius"
          style={{
            position: 'absolute',
            width: lair.spawnRadius * 2,
            height: lair.spawnRadius * 2,
            borderRadius: '50%',
            border: '2px dashed rgba(255, 255, 255, 0.3)',
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            zIndex: -1,
          }}
        />
      )}

      {/* Lair structure */}
      <div
        className="lair-structure"
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: config.appearance.color,
          borderRadius: '8px',
          border: isHovered ? '3px solid #FFD700' : '2px solid #000',
          boxShadow: isHovered
            ? '0 0 20px rgba(255, 215, 0, 0.5)'
            : '0 4px 8px rgba(0, 0, 0, 0.3)',
          transition: 'all 0.2s ease',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Health bar */}
        <div
          className="health-bar-container"
          style={{
            position: 'absolute',
            top: 5,
            left: '10%',
            right: '10%',
            height: 8,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            borderRadius: 4,
            overflow: 'hidden',
          }}
        >
          <div
            className="health-bar"
            style={{
              height: '100%',
              width: `${healthPercentage}%`,
              backgroundColor:
                healthPercentage > 50
                  ? '#4CAF50'
                  : healthPercentage > 25
                  ? '#FFA500'
                  : '#FF0000',
              transition: 'width 0.3s ease, background-color 0.3s ease',
            }}
          />
        </div>

        {/* Lair icon/label */}
        <div
          style={{
            color: '#fff',
            textAlign: 'center',
            fontSize: 12,
            fontWeight: 'bold',
            textShadow: '1px 1px 2px rgba(0, 0, 0, 0.8)',
            userSelect: 'none',
          }}
        >
          {config.displayName}
        </div>
      </div>

      {/* Tooltip */}
      {isHovered && (
        <div
          className="lair-tooltip"
          style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginTop: 10,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: '#fff',
            padding: '8px 12px',
            borderRadius: 4,
            fontSize: 11,
            whiteSpace: 'nowrap',
            zIndex: 1000,
            pointerEvents: 'none',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.5)',
          }}
        >
          <div>
            <strong>{config.displayName}</strong>
          </div>
          <div>
            Health: {lair.health} / {lair.maxHealth}
          </div>
          <div>Max Mobs: {lair.maxMobs}</div>
          <div>Spawn Interval: {config.spawnInterval / 1000}s</div>
        </div>
      )}
    </div>
  );
};

export default MobLair;
