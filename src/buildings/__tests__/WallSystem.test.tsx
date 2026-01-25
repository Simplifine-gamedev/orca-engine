/**
 * Tests for Wall System
 * These tests verify the preloading, caching, and performance improvements
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import WallPreviewCache, {
  initializeWallSystem,
  cleanupWallSystem,
} from '../WallSystem';

describe('WallPreviewCache', () => {
  let cache: WallPreviewCache;

  beforeEach(() => {
    cache = WallPreviewCache.getInstance();
  });

  afterEach(() => {
    cleanupWallSystem();
  });

  it('should be a singleton', () => {
    const cache1 = WallPreviewCache.getInstance();
    const cache2 = WallPreviewCache.getInstance();
    expect(cache1).toBe(cache2);
  });

  it('should preload assets successfully', async () => {
    expect(cache.isAssetsPreloaded()).toBe(false);
    
    await cache.preloadAssets();
    
    expect(cache.isAssetsPreloaded()).toBe(true);
  });

  it('should not preload assets multiple times', async () => {
    await cache.preloadAssets();
    const firstPreload = cache.isAssetsPreloaded();
    
    await cache.preloadAssets();
    const secondPreload = cache.isAssetsPreloaded();
    
    expect(firstPreload).toBe(true);
    expect(secondPreload).toBe(true);
  });

  it('should cache wall geometries', async () => {
    await cache.preloadAssets();
    
    const segmentGeometry = cache.getGeometry('wall_segment');
    const cornerGeometry = cache.getGeometry('wall_corner');
    const gateGeometry = cache.getGeometry('wall_gate');
    
    expect(segmentGeometry).toBeDefined();
    expect(cornerGeometry).toBeDefined();
    expect(gateGeometry).toBeDefined();
  });

  it('should cache wall materials', async () => {
    await cache.preloadAssets();
    
    const previewMaterial = cache.getMaterial('wall_preview');
    const validMaterial = cache.getMaterial('wall_preview_valid');
    const invalidMaterial = cache.getMaterial('wall_preview_invalid');
    
    expect(previewMaterial).toBeDefined();
    expect(validMaterial).toBeDefined();
    expect(invalidMaterial).toBeDefined();
  });

  it('should reuse cached geometries', async () => {
    await cache.preloadAssets();
    
    const geometry1 = cache.getGeometry('wall_segment');
    const geometry2 = cache.getGeometry('wall_segment');
    
    expect(geometry1).toBe(geometry2);
  });

  it('should reuse cached materials', async () => {
    await cache.preloadAssets();
    
    const material1 = cache.getMaterial('wall_preview_valid');
    const material2 = cache.getMaterial('wall_preview_valid');
    
    expect(material1).toBe(material2);
  });

  it('should clean up resources when disposed', async () => {
    await cache.preloadAssets();
    expect(cache.isAssetsPreloaded()).toBe(true);
    
    cache.dispose();
    
    expect(cache.isAssetsPreloaded()).toBe(false);
    expect(cache.getGeometry('wall_segment')).toBeUndefined();
    expect(cache.getMaterial('wall_preview')).toBeUndefined();
  });
});

describe('initializeWallSystem', () => {
  afterEach(() => {
    cleanupWallSystem();
  });

  it('should initialize the wall system', async () => {
    await initializeWallSystem();
    
    const cache = WallPreviewCache.getInstance();
    expect(cache.isAssetsPreloaded()).toBe(true);
  });

  it('should preload assets within reasonable time', async () => {
    const startTime = performance.now();
    
    await initializeWallSystem();
    
    const loadTime = performance.now() - startTime;
    
    // Should load in less than 1 second
    expect(loadTime).toBeLessThan(1000);
  });
});

describe('Performance', () => {
  beforeEach(async () => {
    await initializeWallSystem();
  });

  afterEach(() => {
    cleanupWallSystem();
  });

  it('should retrieve cached geometries quickly', () => {
    const cache = WallPreviewCache.getInstance();
    const iterations = 1000;
    
    const startTime = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      cache.getGeometry('wall_segment');
    }
    
    const avgTime = (performance.now() - startTime) / iterations;
    
    // Average retrieval should be < 0.01ms
    expect(avgTime).toBeLessThan(0.01);
  });

  it('should retrieve cached materials quickly', () => {
    const cache = WallPreviewCache.getInstance();
    const iterations = 1000;
    
    const startTime = performance.now();
    
    for (let i = 0; i < iterations; i++) {
      cache.getMaterial('wall_preview_valid');
    }
    
    const avgTime = (performance.now() - startTime) / iterations;
    
    // Average retrieval should be < 0.01ms
    expect(avgTime).toBeLessThan(0.01);
  });
});
