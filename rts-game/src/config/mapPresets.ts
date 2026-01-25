/**
 * Map Presets Configuration
 * Different map sizes and layouts for Orca RTS
 */

import { MapPreset, MAP_DIMENSIONS } from '../types/MapTypes';

export const MAP_PRESETS: MapPreset[] = [
  // Small Maps
  {
    id: 'small-islands',
    name: 'Scattered Islands',
    description: 'Small archipelago with limited resources. Fast-paced gameplay.',
    size: 'small',
    layout: 'islands',
    width: MAP_DIMENSIONS.small.width,
    height: MAP_DIMENSIONS.small.height,
    maxPlayers: 2,
    thumbnailPath: '/assets/maps/small-islands.png',
    previewColor: '#3B82F6',
    terrain: {
      water: 60,
      land: 35,
      mountains: 5,
    },
    resources: {
      high: false,
      distribution: 'balanced',
    },
  },
  {
    id: 'small-continents',
    name: 'Twin Continents',
    description: 'Two large landmasses separated by ocean. Perfect for 2 players.',
    size: 'small',
    layout: 'continents',
    width: MAP_DIMENSIONS.small.width,
    height: MAP_DIMENSIONS.small.height,
    maxPlayers: 2,
    thumbnailPath: '/assets/maps/small-continents.png',
    previewColor: '#10B981',
    terrain: {
      water: 40,
      land: 50,
      mountains: 10,
    },
    resources: {
      high: true,
      distribution: 'balanced',
    },
  },

  // Medium Maps
  {
    id: 'medium-archipelago',
    name: 'Grand Archipelago',
    description: 'Multiple islands of varying sizes. Naval combat is crucial.',
    size: 'medium',
    layout: 'archipelago',
    width: MAP_DIMENSIONS.medium.width,
    height: MAP_DIMENSIONS.medium.height,
    maxPlayers: 4,
    thumbnailPath: '/assets/maps/medium-archipelago.png',
    previewColor: '#06B6D4',
    terrain: {
      water: 55,
      land: 40,
      mountains: 5,
    },
    resources: {
      high: false,
      distribution: 'clustered',
    },
  },
  {
    id: 'medium-pangaea',
    name: 'Central Pangaea',
    description: 'One large supercontinent. Land-focused gameplay.',
    size: 'medium',
    layout: 'pangaea',
    width: MAP_DIMENSIONS.medium.width,
    height: MAP_DIMENSIONS.medium.height,
    maxPlayers: 4,
    thumbnailPath: '/assets/maps/medium-pangaea.png',
    previewColor: '#22C55E',
    terrain: {
      water: 25,
      land: 65,
      mountains: 10,
    },
    resources: {
      high: true,
      distribution: 'balanced',
    },
  },
  {
    id: 'medium-desert',
    name: 'Desert Oasis',
    description: 'Harsh desert with scattered oases. Resources are scarce.',
    size: 'medium',
    layout: 'desert',
    width: MAP_DIMENSIONS.medium.width,
    height: MAP_DIMENSIONS.medium.height,
    maxPlayers: 3,
    thumbnailPath: '/assets/maps/medium-desert.png',
    previewColor: '#F59E0B',
    terrain: {
      water: 10,
      land: 85,
      mountains: 5,
    },
    resources: {
      high: false,
      distribution: 'clustered',
    },
  },

  // Large Maps
  {
    id: 'large-continents',
    name: 'Four Corners',
    description: 'Four major continents with rich resources.',
    size: 'large',
    layout: 'continents',
    width: MAP_DIMENSIONS.large.width,
    height: MAP_DIMENSIONS.large.height,
    maxPlayers: 6,
    thumbnailPath: '/assets/maps/large-continents.png',
    previewColor: '#14B8A6',
    terrain: {
      water: 45,
      land: 50,
      mountains: 5,
    },
    resources: {
      high: true,
      distribution: 'balanced',
    },
  },
  {
    id: 'large-islands',
    name: 'Island Empire',
    description: 'Numerous islands perfect for naval strategy.',
    size: 'large',
    layout: 'islands',
    width: MAP_DIMENSIONS.large.width,
    height: MAP_DIMENSIONS.large.height,
    maxPlayers: 6,
    thumbnailPath: '/assets/maps/large-islands.png',
    previewColor: '#2563EB',
    terrain: {
      water: 65,
      land: 30,
      mountains: 5,
    },
    resources: {
      high: false,
      distribution: 'random',
    },
  },
  {
    id: 'large-arctic',
    name: 'Frozen Tundra',
    description: 'Ice and snow dominate. Harsh conditions for all.',
    size: 'large',
    layout: 'arctic',
    width: MAP_DIMENSIONS.large.width,
    height: MAP_DIMENSIONS.large.height,
    maxPlayers: 4,
    thumbnailPath: '/assets/maps/large-arctic.png',
    previewColor: '#60A5FA',
    terrain: {
      water: 30,
      land: 60,
      mountains: 10,
    },
    resources: {
      high: false,
      distribution: 'clustered',
    },
  },

  // Huge Maps
  {
    id: 'huge-pangaea',
    name: 'Mega Continent',
    description: 'Massive landmass for epic battles. 8 players recommended.',
    size: 'huge',
    layout: 'pangaea',
    width: MAP_DIMENSIONS.huge.width,
    height: MAP_DIMENSIONS.huge.height,
    maxPlayers: 8,
    thumbnailPath: '/assets/maps/huge-pangaea.png',
    previewColor: '#16A34A',
    terrain: {
      water: 20,
      land: 70,
      mountains: 10,
    },
    resources: {
      high: true,
      distribution: 'balanced',
    },
  },
  {
    id: 'huge-archipelago',
    name: 'Ocean World',
    description: 'Vast ocean with hundreds of islands. Naval supremacy wins.',
    size: 'huge',
    layout: 'archipelago',
    width: MAP_DIMENSIONS.huge.width,
    height: MAP_DIMENSIONS.huge.height,
    maxPlayers: 8,
    thumbnailPath: '/assets/maps/huge-archipelago.png',
    previewColor: '#0284C7',
    terrain: {
      water: 70,
      land: 25,
      mountains: 5,
    },
    resources: {
      high: true,
      distribution: 'random',
    },
  },
];

// Helper functions
export const getMapsBySize = (size: string) => {
  return MAP_PRESETS.filter((map) => map.size === size);
};

export const getMapById = (id: string) => {
  return MAP_PRESETS.find((map) => map.id === id);
};

export const getMapsByMaxPlayers = (playerCount: number) => {
  return MAP_PRESETS.filter((map) => map.maxPlayers >= playerCount);
};
