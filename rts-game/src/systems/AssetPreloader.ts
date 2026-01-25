import * as THREE from 'three';
import { GLTFLoader, GLTF } from 'three/addons/loaders/GLTFLoader.js';

export type AssetType = 'model' | 'texture' | 'audio';

export interface Asset {
  url: string;
  type: AssetType;
  name: string;
  critical?: boolean; // If true, game won't start without this asset
}

export interface LoadingProgress {
  loaded: number;
  total: number;
  currentAsset: string;
  percentage: number;
  assetType: AssetType | null;
}

export interface PreloadedAssets {
  models: Map<string, GLTF>;
  textures: Map<string, THREE.Texture>;
  audio: Map<string, AudioBuffer>;
}

/**
 * GLTF Model Cache - prevents re-downloading same models
 * This is the existing cache mentioned in the issue that Ali added
 */
export const gltfCache = new Map<string, GLTF>();

/**
 * AssetPreloader - Preloads all game assets before starting
 * Fixes the issue where models (especially workers) take too long to load
 */
export class AssetPreloader {
  private assets: Asset[] = [];
  private loadedAssets: PreloadedAssets = {
    models: new Map(),
    textures: new Map(),
    audio: new Map(),
  };
  private gltfLoader: GLTFLoader;
  private textureLoader: THREE.TextureLoader;
  private audioContext: AudioContext | null = null;
  private onProgress?: (progress: LoadingProgress) => void;
  private onComplete?: (assets: PreloadedAssets) => void;
  private onError?: (error: Error, asset: Asset) => void;

  constructor() {
    this.gltfLoader = new GLTFLoader();
    this.textureLoader = new THREE.TextureLoader();
  }

  /**
   * Register assets to preload
   */
  registerAssets(assets: Asset[]): void {
    this.assets.push(...assets);
  }

  /**
   * Set progress callback
   */
  setProgressCallback(callback: (progress: LoadingProgress) => void): void {
    this.onProgress = callback;
  }

  /**
   * Set completion callback
   */
  setCompleteCallback(callback: (assets: PreloadedAssets) => void): void {
    this.onComplete = callback;
  }

  /**
   * Set error callback
   */
  setErrorCallback(callback: (error: Error, asset: Asset) => void): void {
    this.onError = callback;
  }

  /**
   * Start preloading all registered assets
   */
  async preloadAll(): Promise<PreloadedAssets> {
    const total = this.assets.length;
    let loaded = 0;

    console.log(`[AssetPreloader] Starting preload of ${total} assets`);

    for (const asset of this.assets) {
      try {
        await this.loadAsset(asset);
        loaded++;

        // Report progress
        if (this.onProgress) {
          this.onProgress({
            loaded,
            total,
            currentAsset: asset.name,
            percentage: Math.round((loaded / total) * 100),
            assetType: asset.type,
          });
        }

        console.log(`[AssetPreloader] Loaded ${loaded}/${total}: ${asset.name}`);
      } catch (error) {
        console.error(`[AssetPreloader] Failed to load ${asset.name}:`, error);
        
        if (this.onError) {
          this.onError(error as Error, asset);
        }

        // If it's a critical asset, throw error and stop loading
        if (asset.critical) {
          throw new Error(`Critical asset failed to load: ${asset.name}`);
        }
      }
    }

    console.log('[AssetPreloader] All assets loaded successfully');

    if (this.onComplete) {
      this.onComplete(this.loadedAssets);
    }

    return this.loadedAssets;
  }

  /**
   * Load a single asset based on its type
   */
  private async loadAsset(asset: Asset): Promise<void> {
    switch (asset.type) {
      case 'model':
        await this.loadModel(asset);
        break;
      case 'texture':
        await this.loadTexture(asset);
        break;
      case 'audio':
        await this.loadAudio(asset);
        break;
      default:
        throw new Error(`Unknown asset type: ${asset.type}`);
    }
  }

  /**
   * Load a GLTF model with caching
   */
  private async loadModel(asset: Asset): Promise<void> {
    // Check cache first
    if (gltfCache.has(asset.url)) {
      const cached = gltfCache.get(asset.url)!;
      this.loadedAssets.models.set(asset.name, cached);
      console.log(`[AssetPreloader] Using cached model: ${asset.name}`);
      return;
    }

    return new Promise<void>((resolve, reject) => {
      this.gltfLoader.load(
        asset.url,
        (gltf: GLTF) => {
          // Cache the loaded model
          gltfCache.set(asset.url, gltf);
          this.loadedAssets.models.set(asset.name, gltf);
          resolve();
        },
        undefined,
        (error: unknown) => {
          reject(error);
        }
      );
    });
  }

  /**
   * Load a texture
   */
  private async loadTexture(asset: Asset): Promise<void> {
    return new Promise((resolve, reject) => {
      this.textureLoader.load(
        asset.url,
        (texture) => {
          this.loadedAssets.textures.set(asset.name, texture);
          resolve();
        },
        undefined,
        (error) => {
          reject(error);
        }
      );
    });
  }

  /**
   * Load an audio file
   */
  private async loadAudio(asset: Asset): Promise<void> {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }

    const response = await fetch(asset.url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    
    this.loadedAssets.audio.set(asset.name, audioBuffer);
  }

  /**
   * Get a preloaded model
   */
  getModel(name: string): GLTF | undefined {
    return this.loadedAssets.models.get(name);
  }

  /**
   * Get a preloaded texture
   */
  getTexture(name: string): THREE.Texture | undefined {
    return this.loadedAssets.textures.get(name);
  }

  /**
   * Get preloaded audio
   */
  getAudio(name: string): AudioBuffer | undefined {
    return this.loadedAssets.audio.get(name);
  }

  /**
   * Get all preloaded assets
   */
  getAssets(): PreloadedAssets {
    return this.loadedAssets;
  }
}

// Global preloader instance
export const assetPreloader = new AssetPreloader();
