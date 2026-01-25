import React, { useEffect, useState } from 'react';
import { MobLair, MobLairConfig, LAIR_TYPES, LootItem } from '../objects/MobLair';
import { useMobStore } from '../store/mobStore';
import { io, Socket } from 'socket.io-client';

interface GameWorldProps {
  serverUrl?: string;
}

export const GameWorld: React.FC<GameWorldProps> = ({ 
  serverUrl = 'http://localhost:3001' 
}) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const { getAllLairs, addLair, updateLair, getAllMobs, spawnMob, updateMob, killMob } = useMobStore();
  const [loot, setLoot] = useState<Array<{ position: { x: number; y: number }, items: LootItem[] }>>([]);

  useEffect(() => {
    // Connect to game server
    const socketConnection = io(serverUrl);
    setSocket(socketConnection);

    // Handle initial game state
    socketConnection.on('game:state', (state: any) => {
      state.lairs.forEach((lair: MobLairConfig) => {
        addLair(lair);
      });

      state.mobs.forEach((mob: any) => {
        spawnMob(mob);
      });
    });

    // Handle lair events
    socketConnection.on('lair:created', (lair: MobLairConfig) => {
      addLair(lair);
    });

    socketConnection.on('lair:damaged', ({ lairId, health, maxHealth }: any) => {
      updateLair(lairId, { health, maxHealth });
    });

    socketConnection.on('lair:destroyed', ({ lairId, position, loot: droppedLoot }: any) => {
      updateLair(lairId, { isDestroyed: true, health: 0 });
      setLoot(prev => [...prev, { position, items: droppedLoot }]);
    });

    socketConnection.on('lair:removed', ({ lairId }: any) => {
      // Handle lair removal if needed
    });

    // Handle mob events
    socketConnection.on('mob:spawned', (mob: any) => {
      spawnMob(mob);
    });

    socketConnection.on('mob:damaged', ({ mobId, health, maxHealth }: any) => {
      updateMob(mobId, { health, maxHealth });
    });

    socketConnection.on('mob:killed', ({ mobId }: any) => {
      killMob(mobId);
    });

    socketConnection.on('mob:moved', ({ mobId, position }: any) => {
      updateMob(mobId, { position });
    });

    return () => {
      socketConnection.disconnect();
    };
  }, [serverUrl, addLair, updateLair, spawnMob, updateMob, killMob]);

  const handleLairDestroy = (lair: MobLairConfig, droppedLoot: LootItem[]) => {
    console.log(`Lair ${lair.id} destroyed! Loot:`, droppedLoot);
    setLoot(prev => [...prev, { position: lair.position, items: droppedLoot }]);
  };

  const handleMobSpawn = (mobType: string, position: { x: number; y: number }) => {
    console.log(`${mobType} spawned at`, position);
  };

  const handleCreateLair = (type: string, x: number, y: number) => {
    if (socket) {
      socket.emit('lair:create', { type, position: { x, y } });
    }
  };

  const lairs = getAllLairs();
  const mobs = getAllMobs();

  return (
    <div className="game-world" style={{ position: 'relative', width: '100%', height: '100vh', backgroundColor: '#2a4a2a' }}>
      <div className="controls" style={{ position: 'absolute', top: 10, left: 10, zIndex: 1000, backgroundColor: 'rgba(0,0,0,0.7)', padding: '10px', borderRadius: '5px' }}>
        <h3 style={{ color: 'white', marginBottom: '10px' }}>Create Lair</h3>
        {Object.keys(LAIR_TYPES).map(type => (
          <button
            key={type}
            onClick={() => handleCreateLair(type, Math.random() * 800 + 100, Math.random() * 600 + 100)}
            style={{ margin: '5px', padding: '8px 12px', cursor: 'pointer' }}
          >
            {type.replace('_', ' ')}
          </button>
        ))}
      </div>

      <div className="stats" style={{ position: 'absolute', top: 10, right: 10, zIndex: 1000, backgroundColor: 'rgba(0,0,0,0.7)', padding: '10px', borderRadius: '5px', color: 'white' }}>
        <div>Lairs: {lairs.filter(l => !l.isDestroyed).length}/{lairs.length}</div>
        <div>Mobs: {mobs.filter(m => m.isAlive).length}</div>
      </div>

      {/* Render lairs */}
      {lairs.map(lair => (
        <MobLair
          key={lair.id}
          config={lair}
          onDestroy={handleLairDestroy}
          onSpawn={handleMobSpawn}
        />
      ))}

      {/* Render mobs */}
      {mobs.filter(mob => mob.isAlive).map(mob => (
        <div
          key={mob.id}
          className={`mob ${mob.type}`}
          style={{
            position: 'absolute',
            left: mob.position.x,
            top: mob.position.y,
            width: '40px',
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '24px',
            cursor: 'pointer',
          }}
          onClick={() => {
            if (socket) {
              socket.emit('mob:damage', { mobId: mob.id, damage: 30 });
            }
          }}
        >
          {getMobSprite(mob.type)}
          <div 
            className="mob-health-bar" 
            style={{
              position: 'absolute',
              bottom: '-5px',
              width: '40px',
              height: '4px',
              backgroundColor: 'red',
            }}
          >
            <div 
              style={{
                width: `${(mob.health / mob.maxHealth) * 100}%`,
                height: '100%',
                backgroundColor: 'green',
              }}
            />
          </div>
        </div>
      ))}

      {/* Render loot */}
      {loot.map((drop, idx) => (
        <div
          key={idx}
          className="loot-drop"
          style={{
            position: 'absolute',
            left: drop.position.x,
            top: drop.position.y,
            padding: '5px',
            backgroundColor: 'rgba(255, 215, 0, 0.8)',
            borderRadius: '5px',
            fontSize: '12px',
            cursor: 'pointer',
          }}
          onClick={() => {
            setLoot(prev => prev.filter((_, i) => i !== idx));
          }}
        >
          💰 {drop.items.map(item => `${item.item} x${item.quantity}`).join(', ')}
        </div>
      ))}
    </div>
  );
};

function getMobSprite(type: string): string {
  const sprites: Record<string, string> = {
    goblin: '👺',
    ogre: '👹',
    skeleton: '💀',
    dragon: '🐉',
  };
  return sprites[type] || '👾';
}

export default GameWorld;
