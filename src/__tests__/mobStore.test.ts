import { renderHook, act } from '@testing-library/react';
import { useMobStore } from '../store/mobStore';
import { MobLairConfig } from '../objects/MobLair';

describe('MobStore', () => {
  beforeEach(() => {
    const { result } = renderHook(() => useMobStore());
    act(() => {
      result.current.clearAll();
    });
  });

  describe('Lair Operations', () => {
    it('should add a lair', () => {
      const { result } = renderHook(() => useMobStore());
      
      const lair: MobLairConfig = {
        id: 'test_lair_1',
        type: 'goblin_camp',
        position: { x: 100, y: 100 },
        health: 500,
        maxHealth: 500,
        spawnInterval: 30000,
        mobType: 'goblin',
        maxMobs: 8,
        lootTable: [],
        isDestroyed: false,
      };

      act(() => {
        result.current.addLair(lair);
      });

      expect(result.current.getAllLairs()).toHaveLength(1);
      expect(result.current.getLair('test_lair_1')).toEqual(lair);
    });

    it('should remove a lair', () => {
      const { result } = renderHook(() => useMobStore());
      
      const lair: MobLairConfig = {
        id: 'test_lair_1',
        type: 'goblin_camp',
        position: { x: 100, y: 100 },
        health: 500,
        maxHealth: 500,
        spawnInterval: 30000,
        mobType: 'goblin',
        maxMobs: 8,
        lootTable: [],
        isDestroyed: false,
      };

      act(() => {
        result.current.addLair(lair);
        result.current.removeLair('test_lair_1');
      });

      expect(result.current.getAllLairs()).toHaveLength(0);
      expect(result.current.getLair('test_lair_1')).toBeUndefined();
    });

    it('should update a lair', () => {
      const { result } = renderHook(() => useMobStore());
      
      const lair: MobLairConfig = {
        id: 'test_lair_1',
        type: 'goblin_camp',
        position: { x: 100, y: 100 },
        health: 500,
        maxHealth: 500,
        spawnInterval: 30000,
        mobType: 'goblin',
        maxMobs: 8,
        lootTable: [],
        isDestroyed: false,
      };

      act(() => {
        result.current.addLair(lair);
        result.current.updateLair('test_lair_1', { health: 250 });
      });

      const updatedLair = result.current.getLair('test_lair_1');
      expect(updatedLair?.health).toBe(250);
    });

    it('should destroy a lair', () => {
      const { result } = renderHook(() => useMobStore());
      
      const lair: MobLairConfig = {
        id: 'test_lair_1',
        type: 'goblin_camp',
        position: { x: 100, y: 100 },
        health: 500,
        maxHealth: 500,
        spawnInterval: 30000,
        mobType: 'goblin',
        maxMobs: 8,
        lootTable: [],
        isDestroyed: false,
      };

      act(() => {
        result.current.addLair(lair);
        result.current.destroyLair('test_lair_1');
      });

      const destroyedLair = result.current.getLair('test_lair_1');
      expect(destroyedLair?.isDestroyed).toBe(true);
      expect(destroyedLair?.health).toBe(0);
    });
  });

  describe('Mob Operations', () => {
    it('should spawn a mob', () => {
      const { result } = renderHook(() => useMobStore());
      
      const mob = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'test_lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      act(() => {
        result.current.spawnMob(mob);
      });

      expect(result.current.getAllMobs()).toHaveLength(1);
      expect(result.current.getMob('test_mob_1')).toEqual(mob);
    });

    it('should remove a mob', () => {
      const { result } = renderHook(() => useMobStore());
      
      const mob = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'test_lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      act(() => {
        result.current.spawnMob(mob);
        result.current.removeMob('test_mob_1');
      });

      expect(result.current.getAllMobs()).toHaveLength(0);
    });

    it('should get mobs by lair', () => {
      const { result } = renderHook(() => useMobStore());
      
      const mob1 = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      const mob2 = {
        id: 'test_mob_2',
        type: 'goblin',
        lairId: 'lair_1',
        position: { x: 160, y: 160 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      const mob3 = {
        id: 'test_mob_3',
        type: 'ogre',
        lairId: 'lair_2',
        position: { x: 300, y: 300 },
        health: 200,
        maxHealth: 200,
        isAlive: true,
      };

      act(() => {
        result.current.spawnMob(mob1);
        result.current.spawnMob(mob2);
        result.current.spawnMob(mob3);
      });

      const lair1Mobs = result.current.getMobsByLair('lair_1');
      expect(lair1Mobs).toHaveLength(2);
      expect(lair1Mobs.every(m => m.lairId === 'lair_1')).toBe(true);
    });

    it('should get only alive mobs', () => {
      const { result } = renderHook(() => useMobStore());
      
      const mob1 = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      const mob2 = {
        id: 'test_mob_2',
        type: 'goblin',
        lairId: 'lair_1',
        position: { x: 160, y: 160 },
        health: 0,
        maxHealth: 100,
        isAlive: false,
      };

      act(() => {
        result.current.spawnMob(mob1);
        result.current.spawnMob(mob2);
      });

      const aliveMobs = result.current.getAliveMobs();
      expect(aliveMobs).toHaveLength(1);
      expect(aliveMobs[0].isAlive).toBe(true);
    });

    it('should kill a mob', () => {
      const { result } = renderHook(() => useMobStore());
      
      const mob = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      act(() => {
        result.current.spawnMob(mob);
        result.current.killMob('test_mob_1');
      });

      const killedMob = result.current.getMob('test_mob_1');
      expect(killedMob?.isAlive).toBe(false);
      expect(killedMob?.health).toBe(0);
    });
  });

  describe('Cleanup', () => {
    it('should clear all lairs and mobs', () => {
      const { result } = renderHook(() => useMobStore());
      
      const lair: MobLairConfig = {
        id: 'test_lair_1',
        type: 'goblin_camp',
        position: { x: 100, y: 100 },
        health: 500,
        maxHealth: 500,
        spawnInterval: 30000,
        mobType: 'goblin',
        maxMobs: 8,
        lootTable: [],
        isDestroyed: false,
      };

      const mob = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'test_lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      act(() => {
        result.current.addLair(lair);
        result.current.spawnMob(mob);
        result.current.clearAll();
      });

      expect(result.current.getAllLairs()).toHaveLength(0);
      expect(result.current.getAllMobs()).toHaveLength(0);
    });

    it('should remove mobs when lair is removed', () => {
      const { result } = renderHook(() => useMobStore());
      
      const lair: MobLairConfig = {
        id: 'test_lair_1',
        type: 'goblin_camp',
        position: { x: 100, y: 100 },
        health: 500,
        maxHealth: 500,
        spawnInterval: 30000,
        mobType: 'goblin',
        maxMobs: 8,
        lootTable: [],
        isDestroyed: false,
      };

      const mob = {
        id: 'test_mob_1',
        type: 'goblin',
        lairId: 'test_lair_1',
        position: { x: 150, y: 150 },
        health: 100,
        maxHealth: 100,
        isAlive: true,
      };

      act(() => {
        result.current.addLair(lair);
        result.current.spawnMob(mob);
        result.current.removeLair('test_lair_1');
      });

      expect(result.current.getAllMobs()).toHaveLength(0);
    });
  });
});
