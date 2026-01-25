import React, { useEffect, useState } from 'react';
import { useMobStore } from '../store/mobStore';

export interface MobLairConfig {
  id: string;
  type: 'goblin_camp' | 'ogre_cave' | 'undead_crypt' | 'dragon_nest';
  position: { x: number; y: number };
  health: number;
  maxHealth: number;
  spawnInterval: number; // milliseconds
  mobType: string;
  maxMobs: number;
  lootTable: LootItem[];
  isDestroyed: boolean;
}

export interface LootItem {
  item: string;
  quantity: number;
  dropChance: number;
}

export const LAIR_TYPES: Record<string, Partial<MobLairConfig>> = {
  goblin_camp: {
    type: 'goblin_camp',
    maxHealth: 500,
    spawnInterval: 30000, // 30 seconds
    mobType: 'goblin',
    maxMobs: 8,
    lootTable: [
      { item: 'gold', quantity: 50, dropChance: 1.0 },
      { item: 'goblin_dagger', quantity: 1, dropChance: 0.3 },
      { item: 'leather_scraps', quantity: 5, dropChance: 0.7 },
    ],
  },
  ogre_cave: {
    type: 'ogre_cave',
    maxHealth: 1200,
    spawnInterval: 60000, // 60 seconds
    mobType: 'ogre',
    maxMobs: 4,
    lootTable: [
      { item: 'gold', quantity: 150, dropChance: 1.0 },
      { item: 'ogre_club', quantity: 1, dropChance: 0.4 },
      { item: 'thick_hide', quantity: 3, dropChance: 0.8 },
    ],
  },
  undead_crypt: {
    type: 'undead_crypt',
    maxHealth: 800,
    spawnInterval: 25000, // 25 seconds
    mobType: 'skeleton',
    maxMobs: 12,
    lootTable: [
      { item: 'gold', quantity: 80, dropChance: 1.0 },
      { item: 'bone_sword', quantity: 1, dropChance: 0.25 },
      { item: 'soul_essence', quantity: 2, dropChance: 0.6 },
    ],
  },
  dragon_nest: {
    type: 'dragon_nest',
    maxHealth: 3000,
    spawnInterval: 120000, // 120 seconds
    mobType: 'dragon',
    maxMobs: 2,
    lootTable: [
      { item: 'gold', quantity: 500, dropChance: 1.0 },
      { item: 'dragon_scale', quantity: 5, dropChance: 0.9 },
      { item: 'legendary_weapon', quantity: 1, dropChance: 0.1 },
    ],
  },
};

interface MobLairProps {
  config: MobLairConfig;
  onDestroy?: (lair: MobLairConfig, loot: LootItem[]) => void;
  onSpawn?: (mobType: string, position: { x: number; y: number }) => void;
}

export const MobLair: React.FC<MobLairProps> = ({ config, onDestroy, onSpawn }) => {
  const { spawnMob, getMobsByLair, destroyLair } = useMobStore();
  const [lastSpawnTime, setLastSpawnTime] = useState(Date.now());
  const [health, setHealth] = useState(config.health);

  useEffect(() => {
    if (config.isDestroyed) return;

    const spawnTimer = setInterval(() => {
      const currentMobs = getMobsByLair(config.id);
      
      if (currentMobs.length < config.maxMobs) {
        const spawnOffset = {
          x: config.position.x + (Math.random() - 0.5) * 100,
          y: config.position.y + (Math.random() - 0.5) * 100,
        };

        spawnMob({
          id: `${config.id}_mob_${Date.now()}_${Math.random()}`,
          type: config.mobType,
          lairId: config.id,
          position: spawnOffset,
          health: 100,
          maxHealth: 100,
        });

        setLastSpawnTime(Date.now());
        
        if (onSpawn) {
          onSpawn(config.mobType, spawnOffset);
        }
      }
    }, config.spawnInterval);

    return () => clearInterval(spawnTimer);
  }, [config, getMobsByLair, spawnMob, onSpawn]);

  const handleDamage = (damage: number) => {
    const newHealth = Math.max(0, health - damage);
    setHealth(newHealth);

    if (newHealth <= 0 && !config.isDestroyed) {
      handleDestroy();
    }
  };

  const handleDestroy = () => {
    const droppedLoot = config.lootTable
      .filter(loot => Math.random() < loot.dropChance)
      .map(loot => ({
        ...loot,
        quantity: Math.floor(loot.quantity * (0.8 + Math.random() * 0.4)), // 80-120% of base quantity
      }));

    destroyLair(config.id);

    if (onDestroy) {
      onDestroy(config, droppedLoot);
    }
  };

  const healthPercent = (health / config.maxHealth) * 100;
  const currentMobCount = getMobsByLair(config.id).length;

  return (
    <div
      className={`mob-lair ${config.type} ${config.isDestroyed ? 'destroyed' : ''}`}
      style={{
        position: 'absolute',
        left: config.position.x,
        top: config.position.y,
        width: '80px',
        height: '80px',
      }}
      onClick={() => handleDamage(50)} // Example: click to damage
    >
      <div className="lair-sprite">
        {getLairSprite(config.type)}
      </div>
      
      {!config.isDestroyed && (
        <>
          <div className="health-bar">
            <div 
              className="health-fill" 
              style={{ width: `${healthPercent}%` }}
            />
          </div>
          
          <div className="mob-count">
            {currentMobCount}/{config.maxMobs}
          </div>
        </>
      )}

      {config.isDestroyed && (
        <div className="destroyed-marker">💀</div>
      )}
    </div>
  );
};

function getLairSprite(type: string): string {
  const sprites: Record<string, string> = {
    goblin_camp: '⛺',
    ogre_cave: '🕳️',
    undead_crypt: '⚰️',
    dragon_nest: '🏔️',
  };
  return sprites[type] || '🏠';
}

export default MobLair;
