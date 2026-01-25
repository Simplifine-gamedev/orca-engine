import * as THREE from 'three';
import { useEffect, useRef, useState } from 'react';

// Wall preview asset cache
class WallPreviewCache {
  private static instance: WallPreviewCache;
  private geometryCache: Map<string, THREE.BufferGeometry> = new Map();
  private materialCache: Map<string, THREE.Material> = new Map();
  private preloadPromise: Promise<void> | null = null;
  private isPreloaded: boolean = false;

  private constructor() {}

  static getInstance(): WallPreviewCache {
    if (!WallPreviewCache.instance) {
      WallPreviewCache.instance = new WallPreviewCache();
    }
    return WallPreviewCache.instance;
  }

  // Preload all wall preview assets
  async preloadAssets(): Promise<void> {
    if (this.isPreloaded) {
      return;
    }

    if (this.preloadPromise) {
      return this.preloadPromise;
    }

    this.preloadPromise = this._preloadAssets();
    return this.preloadPromise;
  }

  private async _preloadAssets(): Promise<void> {
    console.log('[WallSystem] Starting wall preview asset preloading...');
    const startTime = performance.now();

    try {
      // Preload wall geometries
      await this.preloadGeometries();

      // Preload wall materials
      await this.preloadMaterials();

      this.isPreloaded = true;
      const loadTime = performance.now() - startTime;
      console.log(`[WallSystem] Wall preview assets preloaded in ${loadTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('[WallSystem] Failed to preload wall preview assets:', error);
      throw error;
    }
  }

  private async preloadGeometries(): Promise<void> {
    // Create and cache wall segment geometry
    const wallGeometry = new THREE.BoxGeometry(1, 2, 0.2);
    wallGeometry.computeBoundingBox();
    wallGeometry.computeBoundingSphere();
    this.geometryCache.set('wall_segment', wallGeometry);

    // Create and cache wall corner geometry
    const cornerGeometry = new THREE.BoxGeometry(0.3, 2, 0.3);
    cornerGeometry.computeBoundingBox();
    cornerGeometry.computeBoundingSphere();
    this.geometryCache.set('wall_corner', cornerGeometry);

    // Create and cache wall gate geometry
    const gateGeometry = new THREE.BoxGeometry(1.5, 2.5, 0.2);
    gateGeometry.computeBoundingBox();
    gateGeometry.computeBoundingSphere();
    this.geometryCache.set('wall_gate', gateGeometry);

    console.log('[WallSystem] Geometries preloaded');
  }

  private async preloadMaterials(): Promise<void> {
    // Create and cache preview material (semi-transparent for blueprint mode)
    const previewMaterial = new THREE.MeshStandardMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.6,
      roughness: 0.7,
      metalness: 0.3,
    });
    this.materialCache.set('wall_preview', previewMaterial);

    // Create and cache valid placement material (green)
    const validMaterial = new THREE.MeshStandardMaterial({
      color: 0x44ff44,
      transparent: true,
      opacity: 0.6,
      roughness: 0.7,
      metalness: 0.3,
    });
    this.materialCache.set('wall_preview_valid', validMaterial);

    // Create and cache invalid placement material (red)
    const invalidMaterial = new THREE.MeshStandardMaterial({
      color: 0xff4444,
      transparent: true,
      opacity: 0.6,
      roughness: 0.7,
      metalness: 0.3,
    });
    this.materialCache.set('wall_preview_invalid', invalidMaterial);

    console.log('[WallSystem] Materials preloaded');
  }

  getGeometry(type: string): THREE.BufferGeometry | undefined {
    return this.geometryCache.get(type);
  }

  getMaterial(type: string): THREE.Material | undefined {
    return this.materialCache.get(type);
  }

  isAssetsPreloaded(): boolean {
    return this.isPreloaded;
  }

  // Clean up resources when no longer needed
  dispose(): void {
    this.geometryCache.forEach((geometry) => geometry.dispose());
    this.materialCache.forEach((material) => material.dispose());
    this.geometryCache.clear();
    this.materialCache.clear();
    this.isPreloaded = false;
    this.preloadPromise = null;
  }
}

// Wall preview component props
interface WallPreviewProps {
  position: [number, number, number];
  rotation?: [number, number, number];
  type?: 'wall_segment' | 'wall_corner' | 'wall_gate';
  isValid?: boolean;
  onReady?: () => void;
}

// Wall preview component
export const WallPreview: React.FC<WallPreviewProps> = ({
  position,
  rotation = [0, 0, 0],
  type = 'wall_segment',
  isValid = true,
  onReady,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [isLoading, setIsLoading] = useState(true);
  const cache = WallPreviewCache.getInstance();

  useEffect(() => {
    const loadAssets = async () => {
      try {
        // Ensure assets are preloaded
        if (!cache.isAssetsPreloaded()) {
          await cache.preloadAssets();
        }

        setIsLoading(false);
        onReady?.();
      } catch (error) {
        console.error('[WallPreview] Failed to load assets:', error);
        setIsLoading(false);
      }
    };

    loadAssets();
  }, [cache, onReady]);

  if (isLoading) {
    return null; // Don't render until assets are loaded
  }

  const geometry = cache.getGeometry(type);
  const materialType = isValid ? 'wall_preview_valid' : 'wall_preview_invalid';
  const material = cache.getMaterial(materialType);

  if (!geometry || !material) {
    console.error('[WallPreview] Missing geometry or material from cache');
    return null;
  }

  return (
    <mesh
      ref={meshRef}
      position={position}
      rotation={rotation}
      geometry={geometry}
      material={material}
      castShadow
      receiveShadow
    />
  );
};

// Wall system hook for managing wall building mode
export const useWallSystem = () => {
  const [isBuildMode, setIsBuildMode] = useState(false);
  const [isPreloading, setIsPreloading] = useState(false);
  const [wallPreviews, setWallPreviews] = useState<Array<WallPreviewProps>>([]);
  const cache = WallPreviewCache.getInstance();

  // Preload assets when entering build mode
  const enterBuildMode = async () => {
    setIsPreloading(true);

    try {
      // Preload assets if not already loaded
      if (!cache.isAssetsPreloaded()) {
        await cache.preloadAssets();
      }

      setIsBuildMode(true);
    } catch (error) {
      console.error('[WallSystem] Failed to enter build mode:', error);
    } finally {
      setIsPreloading(false);
    }
  };

  const exitBuildMode = () => {
    setIsBuildMode(false);
    setWallPreviews([]);
  };

  const addWallPreview = (preview: WallPreviewProps) => {
    setWallPreviews((prev) => [...prev, preview]);
  };

  const clearWallPreviews = () => {
    setWallPreviews([]);
  };

  return {
    isBuildMode,
    isPreloading,
    wallPreviews,
    enterBuildMode,
    exitBuildMode,
    addWallPreview,
    clearWallPreviews,
  };
};

// Initialize preloading on game start
export const initializeWallSystem = async (): Promise<void> => {
  const cache = WallPreviewCache.getInstance();
  console.log('[WallSystem] Initializing wall system...');
  
  try {
    await cache.preloadAssets();
    console.log('[WallSystem] Wall system initialized successfully');
  } catch (error) {
    console.error('[WallSystem] Failed to initialize wall system:', error);
    throw error;
  }
};

// Cleanup function
export const cleanupWallSystem = (): void => {
  const cache = WallPreviewCache.getInstance();
  cache.dispose();
  console.log('[WallSystem] Wall system cleaned up');
};

export default WallPreviewCache;
