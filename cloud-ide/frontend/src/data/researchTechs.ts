// Research technologies available in the game
import { ResearchTech } from '../types/game';

export const RESEARCH_TECHS: Record<string, ResearchTech> = {
  // Blacksmith Technologies
  IRON_WEAPONS: {
    id: 'iron_weapons',
    name: 'Iron Weapons',
    description: 'Upgrade melee units with iron weapons. +2 attack damage.',
    cost: { gold: 100, stone: 50 },
    researchTime: 30,
    prerequisites: [],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'stat_boost',
        target: 'melee_units',
        value: 2,
        description: '+2 Attack for all melee units',
      },
    ],
    icon: '/icons/iron_weapons.png',
  },
  
  STEEL_WEAPONS: {
    id: 'steel_weapons',
    name: 'Steel Weapons',
    description: 'Further upgrade melee units with steel weapons. +3 attack damage.',
    cost: { gold: 200, stone: 100 },
    researchTime: 60,
    prerequisites: ['iron_weapons'],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'stat_boost',
        target: 'melee_units',
        value: 3,
        description: '+3 Attack for all melee units',
      },
    ],
    icon: '/icons/steel_weapons.png',
  },
  
  LEATHER_ARMOR: {
    id: 'leather_armor',
    name: 'Leather Armor',
    description: 'Equip units with leather armor. +1 armor.',
    cost: { gold: 80, wood: 60 },
    researchTime: 25,
    prerequisites: [],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'stat_boost',
        target: 'all_units',
        value: 1,
        description: '+1 Armor for all units',
      },
    ],
    icon: '/icons/leather_armor.png',
  },
  
  CHAIN_MAIL: {
    id: 'chain_mail',
    name: 'Chain Mail',
    description: 'Upgrade to chain mail armor. +2 armor.',
    cost: { gold: 150, stone: 80 },
    researchTime: 45,
    prerequisites: ['leather_armor'],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'stat_boost',
        target: 'all_units',
        value: 2,
        description: '+2 Armor for all units',
      },
    ],
    icon: '/icons/chain_mail.png',
  },
  
  PLATE_ARMOR: {
    id: 'plate_armor',
    name: 'Plate Armor',
    description: 'The finest armor available. +3 armor.',
    cost: { gold: 300, stone: 150 },
    researchTime: 90,
    prerequisites: ['chain_mail'],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'stat_boost',
        target: 'all_units',
        value: 3,
        description: '+3 Armor for all units',
      },
    ],
    icon: '/icons/plate_armor.png',
  },
  
  FORGING_TECHNIQUES: {
    id: 'forging_techniques',
    name: 'Advanced Forging',
    description: 'Improve forging techniques. Reduces blacksmith research time by 25%.',
    cost: { gold: 150, stone: 100 },
    researchTime: 50,
    prerequisites: [],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'resource_bonus',
        target: 'blacksmith_research_speed',
        value: 0.25,
        description: '-25% Research time for blacksmith technologies',
      },
    ],
    icon: '/icons/forging.png',
  },
  
  SIEGE_WEAPONS: {
    id: 'siege_weapons',
    name: 'Siege Engineering',
    description: 'Unlock siege weapons construction. Enables building siege units.',
    cost: { gold: 250, wood: 200 },
    researchTime: 75,
    prerequisites: ['steel_weapons'],
    buildingRequired: 'blacksmith',
    effects: [
      {
        type: 'unlock_unit',
        target: 'catapult',
        value: 'catapult',
        description: 'Unlocks Catapult unit',
      },
      {
        type: 'unlock_unit',
        target: 'ballista',
        value: 'ballista',
        description: 'Unlocks Ballista unit',
      },
    ],
    icon: '/icons/siege_weapons.png',
  },
};

export const getResearchTech = (id: string): ResearchTech | undefined => {
  return RESEARCH_TECHS[id];
};

export const getBlacksmithTechs = (): ResearchTech[] => {
  return Object.values(RESEARCH_TECHS).filter(
    tech => tech.buildingRequired === 'blacksmith'
  );
};

export const canResearch = (
  tech: ResearchTech,
  completedResearch: string[],
  playerResources: { gold?: number; wood?: number; stone?: number; food?: number }
): boolean => {
  // Check prerequisites
  const hasPrerequisites = tech.prerequisites.every(prereq =>
    completedResearch.includes(prereq)
  );
  
  if (!hasPrerequisites) return false;
  
  // Check resources
  if (tech.cost.gold && (playerResources.gold || 0) < tech.cost.gold) return false;
  if (tech.cost.wood && (playerResources.wood || 0) < tech.cost.wood) return false;
  if (tech.cost.stone && (playerResources.stone || 0) < tech.cost.stone) return false;
  if (tech.cost.food && (playerResources.food || 0) < tech.cost.food) return false;
  
  return true;
};
