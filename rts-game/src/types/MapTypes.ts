/**
 * Map Types and Interfaces for Orca RTS
 */

export type MapSize = 'small' | 'medium' | 'large' | 'huge';
export type MapLayout = 'islands' | 'continents' | 'pangaea' | 'archipelago' | 'desert' | 'arctic';

export interface MapPreset {
  id: string;
  name: string;
  description: string;
  size: MapSize;
  layout: MapLayout;
  width: number;
  height: number;
  maxPlayers: number;
  thumbnailPath: string;
  previewColor: string;
  terrain: {
    water: number;
    land: number;
    mountains: number;
  };
  resources: {
    high: boolean;
    distribution: 'balanced' | 'clustered' | 'random';
  };
}

export interface MapDimensions {
  small: { width: number; height: number };
  medium: { width: number; height: number };
  large: { width: number; height: number };
  huge: { width: number; height: number };
}

export const MAP_DIMENSIONS: MapDimensions = {
  small: { width: 128, height: 128 },
  medium: { width: 256, height: 256 },
  large: { width: 512, height: 512 },
  huge: { width: 1024, height: 1024 },
};
