import React, { useEffect } from 'react';
import { useMobStore } from '../store/mobStore';
import { LAIR_TYPES, MobLair } from '../objects/MobLair';

/**
 * Advanced example showing how to create and manage lairs programmatically
 * without using the GameWorld component.
 */
export const CustomLairExample: React.FC = () => {
  const { addLair, getAllLairs, getMobsByLair } = useMobStore();

  useEffect(() => {
    // Create multiple lairs on mount
    const lairs = [
      {
        id: 'custom_goblin_1',
        ...LAIR_TYPES.goblin_camp,
        position: { x: 200, y: 200 },
        health: 500,
        isDestroyed: false,
      },
      {
        id: 'custom_ogre_1',
        ...LAIR_TYPES.ogre_cave,
        position: { x: 600, y: 300 },
        health: 1200,
        isDestroyed: false,
      },
      {
        id: 'custom_dragon_1',
        ...LAIR_TYPES.dragon_nest,
        position: { x: 400, y: 500 },
        health: 3000,
        isDestroyed: false,
      },
    ];

    lairs.forEach(lair => addLair(lair as any));
  }, [addLair]);

  const handleLairDestroy = (lair: any, loot: any) => {
    console.log('Lair destroyed:', lair.id);
    console.log('Loot dropped:', loot);
    alert(`Lair destroyed! You received: ${loot.map((l: any) => `${l.item} x${l.quantity}`).join(', ')}`);
  };

  const handleMobSpawn = (mobType: string, position: { x: number; y: number }) => {
    console.log(`${mobType} spawned at`, position);
  };

  const lairs = getAllLairs();

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', backgroundColor: '#1a1a1a' }}>
      <div style={{ position: 'absolute', top: 10, left: 10, color: 'white', backgroundColor: 'rgba(0,0,0,0.8)', padding: '10px', borderRadius: '5px', zIndex: 1000 }}>
        <h2>Custom Lair Management</h2>
        <div>Total Lairs: {lairs.length}</div>
        <div>Active Lairs: {lairs.filter(l => !l.isDestroyed).length}</div>
        {lairs.map(lair => (
          <div key={lair.id} style={{ marginTop: '5px', fontSize: '12px' }}>
            {lair.type}: {lair.health}/{lair.maxHealth} HP - {getMobsByLair(lair.id).length} mobs
          </div>
        ))}
      </div>

      {lairs.map(lair => (
        <MobLair
          key={lair.id}
          config={lair}
          onDestroy={handleLairDestroy}
          onSpawn={handleMobSpawn}
        />
      ))}
    </div>
  );
};

export default CustomLairExample;
