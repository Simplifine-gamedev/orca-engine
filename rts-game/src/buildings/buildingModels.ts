import { BuildingModel, BuildingType } from '../types/building';
import { ResearchItem, ResearchTech } from '../types/research';

export const buildingModels: Record<BuildingType, BuildingModel> = {
  headquarters: {
    id: 'headquarters',
    name: 'Headquarters',
    description: 'Main building for your base. Trains villagers.',
    cost: { wood: 0, stone: 0, gold: 0 },
    buildTime: 0,
    size: { width: 4, height: 4 },
    modelPath: '/models/buildings/headquarters.glb',
    thumbnail: '/thumbnails/headquarters.png',
    hitPoints: 2000,
    canProduce: ['villager'],
  },
  
  barracks: {
    id: 'barracks',
    name: 'Barracks',
    description: 'Trains infantry units.',
    cost: { wood: 150, stone: 0, gold: 0 },
    buildTime: 50,
    size: { width: 3, height: 3 },
    modelPath: '/models/buildings/barracks.glb',
    thumbnail: '/thumbnails/barracks.png',
    hitPoints: 1200,
    canProduce: ['swordsman', 'spearman', 'militia'],
  },
  
  blacksmith: {
    id: 'blacksmith',
    name: 'Blacksmith',
    description: 'Research facility for improving weapons and armor. Essential for military upgrades.',
    cost: { wood: 200, stone: 100, gold: 0 },
    buildTime: 60,
    size: { width: 2, height: 2 },
    modelPath: '/models/buildings/blacksmith.glb',
    thumbnail: '/thumbnails/blacksmith.png',
    hitPoints: 800,
    canResearch: [
      'iron_weapons',
      'steel_armor',
      'advanced_metallurgy',
      'weapon_sharpening',
      'armor_reinforcement',
      'blacksmithing_mastery',
    ],
  },
  
  archery_range: {
    id: 'archery_range',
    name: 'Archery Range',
    description: 'Trains ranged units.',
    cost: { wood: 175, stone: 0, gold: 0 },
    buildTime: 50,
    size: { width: 3, height: 2 },
    modelPath: '/models/buildings/archery_range.glb',
    thumbnail: '/thumbnails/archery_range.png',
    hitPoints: 1000,
    canProduce: ['archer', 'crossbowman'],
  },
  
  stable: {
    id: 'stable',
    name: 'Stable',
    description: 'Trains cavalry units.',
    cost: { wood: 200, stone: 0, gold: 0 },
    buildTime: 60,
    size: { width: 3, height: 3 },
    modelPath: '/models/buildings/stable.glb',
    thumbnail: '/thumbnails/stable.png',
    hitPoints: 1200,
    canProduce: ['knight', 'scout'],
  },
  
  farm: {
    id: 'farm',
    name: 'Farm',
    description: 'Produces food for your civilization.',
    cost: { wood: 60, stone: 0, gold: 0 },
    buildTime: 20,
    size: { width: 2, height: 2 },
    modelPath: '/models/buildings/farm.glb',
    thumbnail: '/thumbnails/farm.png',
    hitPoints: 400,
  },
  
  mine: {
    id: 'mine',
    name: 'Mine',
    description: 'Extracts stone and gold from deposits.',
    cost: { wood: 100, stone: 0, gold: 0 },
    buildTime: 40,
    size: { width: 2, height: 2 },
    modelPath: '/models/buildings/mine.glb',
    thumbnail: '/thumbnails/mine.png',
    hitPoints: 600,
  },
  
  lumbermill: {
    id: 'lumbermill',
    name: 'Lumbermill',
    description: 'Processes wood more efficiently.',
    cost: { wood: 100, stone: 0, gold: 0 },
    buildTime: 35,
    size: { width: 2, height: 2 },
    modelPath: '/models/buildings/lumbermill.glb',
    thumbnail: '/thumbnails/lumbermill.png',
    hitPoints: 600,
  },
};

export const blacksmithResearch: Record<ResearchTech, ResearchItem> = {
  iron_weapons: {
    id: 'iron_weapons',
    name: 'Iron Weapons',
    description: 'Forge weapons from iron, increasing melee attack damage.',
    icon: '/icons/research/iron_weapons.png',
    cost: { gold: 100, food: 50 },
    researchTime: 30,
    requirements: [],
    effects: {
      description: '+2 attack for all melee units',
      modifier: {
        type: 'attack',
        value: 2,
        unit: 'all_melee',
      },
    },
    building: 'blacksmith',
  },
  
  steel_armor: {
    id: 'steel_armor',
    name: 'Steel Armor',
    description: 'Craft steel armor to improve unit defense.',
    icon: '/icons/research/steel_armor.png',
    cost: { gold: 150, food: 75 },
    researchTime: 40,
    requirements: ['iron_weapons'],
    effects: {
      description: '+1 armor for all units',
      modifier: {
        type: 'defense',
        value: 1,
        unit: 'all',
      },
    },
    building: 'blacksmith',
  },
  
  advanced_metallurgy: {
    id: 'advanced_metallurgy',
    name: 'Advanced Metallurgy',
    description: 'Master advanced metalworking techniques.',
    icon: '/icons/research/advanced_metallurgy.png',
    cost: { gold: 300, food: 150 },
    researchTime: 60,
    requirements: ['iron_weapons', 'steel_armor'],
    effects: {
      description: '+3 attack and +2 armor for all melee units',
      modifier: {
        type: 'attack',
        value: 3,
        unit: 'all_melee',
      },
    },
    building: 'blacksmith',
  },
  
  weapon_sharpening: {
    id: 'weapon_sharpening',
    name: 'Weapon Sharpening',
    description: 'Sharpen weapons to increase their effectiveness.',
    icon: '/icons/research/weapon_sharpening.png',
    cost: { gold: 80, food: 40 },
    researchTime: 25,
    requirements: [],
    effects: {
      description: '+1 attack for all units',
      modifier: {
        type: 'attack',
        value: 1,
        unit: 'all',
      },
    },
    building: 'blacksmith',
  },
  
  armor_reinforcement: {
    id: 'armor_reinforcement',
    name: 'Armor Reinforcement',
    description: 'Reinforce armor with additional plating.',
    icon: '/icons/research/armor_reinforcement.png',
    cost: { gold: 120, food: 60 },
    researchTime: 35,
    requirements: ['steel_armor'],
    effects: {
      description: '+2 armor for all units',
      modifier: {
        type: 'defense',
        value: 2,
        unit: 'all',
      },
    },
    building: 'blacksmith',
  },
  
  blacksmithing_mastery: {
    id: 'blacksmithing_mastery',
    name: 'Blacksmithing Mastery',
    description: 'Achieve mastery in blacksmithing, reducing production costs.',
    icon: '/icons/research/blacksmithing_mastery.png',
    cost: { gold: 400, food: 200 },
    researchTime: 90,
    requirements: ['advanced_metallurgy', 'armor_reinforcement'],
    effects: {
      description: '-20% cost for all military units',
      modifier: {
        type: 'cost_reduction',
        value: 20,
        unit: 'all_military',
      },
    },
    building: 'blacksmith',
  },
};

export function getBuildingModel(type: BuildingType): BuildingModel {
  return buildingModels[type];
}

export function getBlacksmithResearch(): ResearchItem[] {
  return Object.values(blacksmithResearch);
}

export function canResearch(techId: ResearchTech, completedResearch: ResearchTech[]): boolean {
  const tech = blacksmithResearch[techId];
  if (!tech.requirements || tech.requirements.length === 0) {
    return true;
  }
  return tech.requirements.every(req => completedResearch.includes(req));
}
