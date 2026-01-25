export interface MapPreset {
  id: string;
  name: string;
  description: string;
  thumbnail: string;
  size: {
    width: number;
    height: number;
  };
  maxPlayers: number;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  terrain: string;
  layout: string;
}

export const MAP_PRESETS: MapPreset[] = [
  {
    id: 'coastal_bay',
    name: 'Coastal Bay',
    description: 'A scenic coastal map with naval opportunities and island outposts.',
    thumbnail: '/assets/maps/coastal_bay.png',
    size: { width: 128, height: 128 },
    maxPlayers: 4,
    difficulty: 'Easy',
    terrain: 'mixed',
    layout: 'symmetrical',
  },
  {
    id: 'desert_valley',
    name: 'Desert Valley',
    description: 'Wide open desert with limited resources. Control the center oasis for strategic advantage.',
    thumbnail: '/assets/maps/desert_valley.png',
    size: { width: 96, height: 96 },
    maxPlayers: 2,
    difficulty: 'Medium',
    terrain: 'desert',
    layout: 'mirrored',
  },
  {
    id: 'frozen_wastes',
    name: 'Frozen Wastes',
    description: 'Harsh arctic environment with challenging terrain and scarce resources.',
    thumbnail: '/assets/maps/frozen_wastes.png',
    size: { width: 160, height: 160 },
    maxPlayers: 6,
    difficulty: 'Hard',
    terrain: 'snow',
    layout: 'random',
  },
  {
    id: 'volcanic_crater',
    name: 'Volcanic Crater',
    description: 'Fight around an active volcano with lava flows and geothermal vents.',
    thumbnail: '/assets/maps/volcanic_crater.png',
    size: { width: 112, height: 112 },
    maxPlayers: 4,
    difficulty: 'Hard',
    terrain: 'volcanic',
    layout: 'circular',
  },
  {
    id: 'green_highlands',
    name: 'Green Highlands',
    description: 'Rolling hills with abundant resources. Perfect for new players.',
    thumbnail: '/assets/maps/green_highlands.png',
    size: { width: 80, height: 80 },
    maxPlayers: 3,
    difficulty: 'Easy',
    terrain: 'grass',
    layout: 'triangular',
  },
  {
    id: 'urban_ruins',
    name: 'Urban Ruins',
    description: 'Fight through the remains of a destroyed city. Dense cover and close quarters.',
    thumbnail: '/assets/maps/urban_ruins.png',
    size: { width: 144, height: 144 },
    maxPlayers: 8,
    difficulty: 'Medium',
    terrain: 'urban',
    layout: 'grid',
  },
];
