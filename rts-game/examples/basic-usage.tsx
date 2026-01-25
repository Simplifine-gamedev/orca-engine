import React, { useEffect, useState } from 'react';
import { useMobStore } from '../src/store/mobStore';
import MobLair, { LairType } from '../src/objects/MobLair';

/**
 * Basic example showing how to use the mob lair spawning system
 */
export function BasicLairExample() {
  const {
    createLair,
    destroyLair,
    damageLair,
    getAllLairs,
    getAllMobs,
    startSpawning,
    stopSpawning,
  } = useMobStore();

  const [lairs, setLairs] = useState<any[]>([]);
  const [mobs, setMobs] = useState<any[]>([]);

  useEffect(() => {
    // Create initial lairs
    createLair(LairType.GOBLIN_CAMP, { x: 200, y: 200 });
    createLair(LairType.WOLF_DEN, { x: 400, y: 300 });
    createLair(LairType.OGRE_CAVE, { x: 600, y: 200 });

    // Start spawning
    startSpawning();

    // Update UI every second
    const interval = setInterval(() => {
      setLairs(getAllLairs());
      setMobs(getAllMobs());
    }, 1000);

    return () => {
      clearInterval(interval);
      stopSpawning();
    };
  }, []);

  const handleLairClick = (lairId: string) => {
    console.log(`Clicked lair: ${lairId}`);
    // Example: Attack the lair
    const destroyed = damageLair(lairId, 100);
    if (destroyed) {
      console.log(`Lair ${lairId} destroyed!`);
    }
  };

  const handleDestroyLair = (lairId: string) => {
    const loot = destroyLair(lairId);
    console.log(`Lair destroyed! Loot:`, loot);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '600px', background: '#2a2a2a' }}>
      {/* Render lairs */}
      {lairs.map((lair) => (
        <MobLair
          key={lair.id}
          lair={lair}
          onClick={handleLairClick}
          onDestroy={handleDestroyLair}
        />
      ))}

      {/* Render mobs as simple dots */}
      {mobs.map((mob) => (
        <div
          key={mob.id}
          style={{
            position: 'absolute',
            left: mob.position.x,
            top: mob.position.y,
            width: 20,
            height: 20,
            borderRadius: '50%',
            backgroundColor: '#ff4444',
            border: '2px solid #fff',
            transform: 'translate(-50%, -50%)',
            cursor: 'pointer',
          }}
          title={`${mob.type} (Lv ${mob.level})`}
        />
      ))}

      {/* Stats panel */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          padding: 15,
          background: 'rgba(0, 0, 0, 0.8)',
          color: '#fff',
          borderRadius: 8,
          fontSize: 14,
        }}
      >
        <h3 style={{ margin: '0 0 10px 0' }}>Game Stats</h3>
        <div>Active Lairs: {lairs.length}</div>
        <div>Total Mobs: {mobs.length}</div>
        {lairs.map((lair) => {
          const lairMobs = mobs.filter((m) => m.lairId === lair.id);
          return (
            <div key={lair.id} style={{ marginTop: 5, fontSize: 12 }}>
              {lair.type}: {lairMobs.length}/{lair.maxMobs} mobs
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default BasicLairExample;
