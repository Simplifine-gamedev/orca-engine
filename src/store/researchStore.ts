/**
 * Research Store - Technology tree and research system for Orca RTS
 * Provides meaningful upgrades and progression throughout the game
 */

import { gameStore, Resources, BuildingType, UnitType } from './gameStore';

export enum ResearchCategory {
  ECONOMY = 'economy',
  MILITARY = 'military',
  TECHNOLOGY = 'technology',
  MAGIC = 'magic',
  SPECIAL = 'special',
}

export interface Research {
  id: string;
  name: string;
  description: string;
  category: ResearchCategory;
  cost: Resources;
  researchTime: number; // in seconds
  prerequisites: string[]; // research IDs that must be completed first
  effects: ResearchEffect[];
  icon?: string;
}

export interface ResearchEffect {
  type: 'resource_rate' | 'unit_stat' | 'building_stat' | 'unlock' | 'special_ability';
  target?: string;
  value: number | string;
  description: string;
}

export interface ActiveResearch {
  researchId: string;
  startTime: number;
  endTime: number;
}

// Complete research tree
export const RESEARCH_TREE: Research[] = [
  // Economy Research
  {
    id: 'mining_1',
    name: 'Improved Mining',
    description: 'Workers mine stone and gold 15% faster',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 150, wood: 100, stone: 0, food: 0 },
    researchTime: 30,
    prerequisites: [],
    effects: [
      { type: 'resource_rate', target: 'stone', value: 0.15, description: '+15% stone gathering' },
      { type: 'resource_rate', target: 'gold', value: 0.15, description: '+15% gold gathering' },
    ],
  },
  {
    id: 'mining_2',
    name: 'Advanced Mining',
    description: 'Workers mine stone and gold 30% faster',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 400, wood: 300, stone: 0, food: 0 },
    researchTime: 60,
    prerequisites: ['mining_1'],
    effects: [
      { type: 'resource_rate', target: 'stone', value: 0.30, description: '+30% stone gathering' },
      { type: 'resource_rate', target: 'gold', value: 0.30, description: '+30% gold gathering' },
    ],
  },
  {
    id: 'mining_3',
    name: 'Master Mining',
    description: 'Workers mine stone and gold 50% faster',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 1000, wood: 800, stone: 0, food: 0 },
    researchTime: 120,
    prerequisites: ['mining_2'],
    effects: [
      { type: 'resource_rate', target: 'stone', value: 0.50, description: '+50% stone gathering' },
      { type: 'resource_rate', target: 'gold', value: 0.50, description: '+50% gold gathering' },
    ],
  },
  {
    id: 'forestry_1',
    name: 'Improved Forestry',
    description: 'Workers gather wood 20% faster',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 100, wood: 0, stone: 50, food: 0 },
    researchTime: 30,
    prerequisites: [],
    effects: [
      { type: 'resource_rate', target: 'wood', value: 0.20, description: '+20% wood gathering' },
    ],
  },
  {
    id: 'forestry_2',
    name: 'Advanced Forestry',
    description: 'Workers gather wood 40% faster',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 300, wood: 0, stone: 200, food: 0 },
    researchTime: 60,
    prerequisites: ['forestry_1'],
    effects: [
      { type: 'resource_rate', target: 'wood', value: 0.40, description: '+40% wood gathering' },
    ],
  },
  {
    id: 'agriculture_1',
    name: 'Crop Rotation',
    description: 'Farms produce 25% more food',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 200, wood: 150, stone: 0, food: 0 },
    researchTime: 45,
    prerequisites: [],
    effects: [
      { type: 'resource_rate', target: 'food', value: 0.25, description: '+25% food production' },
    ],
  },
  {
    id: 'agriculture_2',
    name: 'Irrigation',
    description: 'Farms produce 50% more food',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 500, wood: 400, stone: 0, food: 0 },
    researchTime: 90,
    prerequisites: ['agriculture_1'],
    effects: [
      { type: 'resource_rate', target: 'food', value: 0.50, description: '+50% food production' },
    ],
  },
  {
    id: 'market_trade',
    name: 'Trade Routes',
    description: 'Passive gold income +5/minute',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 300, wood: 200, stone: 100, food: 0 },
    researchTime: 60,
    prerequisites: [],
    effects: [
      { type: 'special_ability', value: 'trade_income', description: 'Generate 5 gold per minute' },
    ],
  },
  {
    id: 'coinage',
    name: 'Coinage',
    description: 'All resources can be converted to gold at the market',
    category: ResearchCategory.ECONOMY,
    cost: { gold: 500, wood: 0, stone: 0, food: 0 },
    researchTime: 75,
    prerequisites: ['market_trade'],
    effects: [
      { type: 'unlock', value: 'resource_conversion', description: 'Unlock resource trading' },
    ],
  },

  // Military Research
  {
    id: 'infantry_combat_1',
    name: 'Infantry Combat Training',
    description: 'Swordsmen +2 attack, +1 defense',
    category: ResearchCategory.MILITARY,
    cost: { gold: 250, wood: 0, stone: 150, food: 100 },
    researchTime: 45,
    prerequisites: [],
    effects: [
      { type: 'unit_stat', target: UnitType.SWORDSMAN, value: 2, description: 'Attack +2' },
      { type: 'unit_stat', target: UnitType.SWORDSMAN, value: 1, description: 'Defense +1' },
    ],
  },
  {
    id: 'infantry_combat_2',
    name: 'Advanced Infantry Tactics',
    description: 'Swordsmen +4 attack, +2 defense',
    category: ResearchCategory.MILITARY,
    cost: { gold: 600, wood: 0, stone: 400, food: 200 },
    researchTime: 90,
    prerequisites: ['infantry_combat_1'],
    effects: [
      { type: 'unit_stat', target: UnitType.SWORDSMAN, value: 4, description: 'Attack +4' },
      { type: 'unit_stat', target: UnitType.SWORDSMAN, value: 2, description: 'Defense +2' },
    ],
  },
  {
    id: 'archery_1',
    name: 'Fletching',
    description: 'Archers +2 attack, +10% range',
    category: ResearchCategory.MILITARY,
    cost: { gold: 200, wood: 150, stone: 0, food: 100 },
    researchTime: 40,
    prerequisites: [],
    effects: [
      { type: 'unit_stat', target: UnitType.ARCHER, value: 2, description: 'Attack +2' },
      { type: 'unit_stat', target: UnitType.ARCHER, value: 10, description: 'Range +10%' },
    ],
  },
  {
    id: 'archery_2',
    name: 'Bodkin Arrows',
    description: 'Archers +5 attack, +20% range',
    category: ResearchCategory.MILITARY,
    cost: { gold: 500, wood: 400, stone: 0, food: 200 },
    researchTime: 80,
    prerequisites: ['archery_1'],
    effects: [
      { type: 'unit_stat', target: UnitType.ARCHER, value: 5, description: 'Attack +5' },
      { type: 'unit_stat', target: UnitType.ARCHER, value: 20, description: 'Range +20%' },
    ],
  },
  {
    id: 'cavalry_training',
    name: 'Cavalry Training',
    description: 'Cavalry +3 attack, +15% speed',
    category: ResearchCategory.MILITARY,
    cost: { gold: 400, wood: 0, stone: 200, food: 300 },
    researchTime: 70,
    prerequisites: [],
    effects: [
      { type: 'unit_stat', target: UnitType.CAVALRY, value: 3, description: 'Attack +3' },
      { type: 'unit_stat', target: UnitType.CAVALRY, value: 15, description: 'Speed +15%' },
    ],
  },
  {
    id: 'siege_engineering',
    name: 'Siege Engineering',
    description: 'Unlock siege weapons, +50% damage to buildings',
    category: ResearchCategory.MILITARY,
    cost: { gold: 800, wood: 600, stone: 400, food: 0 },
    researchTime: 100,
    prerequisites: [],
    effects: [
      { type: 'unlock', value: UnitType.SIEGE_ENGINE, description: 'Unlock siege weapons' },
      { type: 'unit_stat', target: UnitType.SIEGE_ENGINE, value: 50, description: '+50% building damage' },
    ],
  },
  {
    id: 'fortifications',
    name: 'Fortifications',
    description: 'All defensive buildings +100 HP, +2 armor',
    category: ResearchCategory.MILITARY,
    cost: { gold: 500, wood: 300, stone: 500, food: 0 },
    researchTime: 90,
    prerequisites: [],
    effects: [
      { type: 'building_stat', target: BuildingType.DEFENSE_TOWER, value: 100, description: 'HP +100' },
      { type: 'building_stat', target: BuildingType.DEFENSE_TOWER, value: 2, description: 'Armor +2' },
    ],
  },

  // Technology Research
  {
    id: 'metallurgy',
    name: 'Metallurgy',
    description: 'All melee units +2 attack',
    category: ResearchCategory.TECHNOLOGY,
    cost: { gold: 600, wood: 0, stone: 400, food: 0 },
    researchTime: 80,
    prerequisites: [],
    effects: [
      { type: 'unit_stat', target: UnitType.SWORDSMAN, value: 2, description: 'Attack +2' },
      { type: 'unit_stat', target: UnitType.CAVALRY, value: 2, description: 'Attack +2' },
    ],
  },
  {
    id: 'plate_armor',
    name: 'Plate Armor',
    description: 'All units +2 defense',
    category: ResearchCategory.TECHNOLOGY,
    cost: { gold: 700, wood: 0, stone: 500, food: 0 },
    researchTime: 90,
    prerequisites: ['metallurgy'],
    effects: [
      { type: 'unit_stat', target: 'all', value: 2, description: 'Defense +2 for all units' },
    ],
  },
  {
    id: 'ballistics',
    name: 'Ballistics',
    description: 'Ranged units +15% accuracy, +1 attack',
    category: ResearchCategory.TECHNOLOGY,
    cost: { gold: 500, wood: 300, stone: 200, food: 0 },
    researchTime: 70,
    prerequisites: [],
    effects: [
      { type: 'unit_stat', target: UnitType.ARCHER, value: 1, description: 'Attack +1' },
      { type: 'unit_stat', target: UnitType.ARCHER, value: 15, description: 'Accuracy +15%' },
    ],
  },
  {
    id: 'engineering',
    name: 'Engineering',
    description: 'Buildings constructed 25% faster',
    category: ResearchCategory.TECHNOLOGY,
    cost: { gold: 400, wood: 300, stone: 200, food: 0 },
    researchTime: 60,
    prerequisites: [],
    effects: [
      { type: 'building_stat', target: 'all', value: 25, description: 'Construction speed +25%' },
    ],
  },
  {
    id: 'architecture',
    name: 'Architecture',
    description: 'Buildings +200 HP, -10% construction cost',
    category: ResearchCategory.TECHNOLOGY,
    cost: { gold: 800, wood: 600, stone: 400, food: 0 },
    researchTime: 100,
    prerequisites: ['engineering'],
    effects: [
      { type: 'building_stat', target: 'all', value: 200, description: 'HP +200' },
      { type: 'building_stat', target: 'all', value: -10, description: 'Cost -10%' },
    ],
  },

  // Magic Research
  {
    id: 'basic_magic',
    name: 'Basic Magic',
    description: 'Unlock mage units and basic spells',
    category: ResearchCategory.MAGIC,
    cost: { gold: 1000, wood: 0, stone: 0, food: 500 },
    researchTime: 120,
    prerequisites: [],
    effects: [
      { type: 'unlock', value: UnitType.MAGE, description: 'Unlock mage units' },
      { type: 'unlock', value: 'fireball', description: 'Unlock Fireball spell' },
    ],
  },
  {
    id: 'advanced_magic',
    name: 'Advanced Magic',
    description: 'Mages +10 attack, unlock advanced spells',
    category: ResearchCategory.MAGIC,
    cost: { gold: 2000, wood: 0, stone: 0, food: 800 },
    researchTime: 180,
    prerequisites: ['basic_magic'],
    effects: [
      { type: 'unit_stat', target: UnitType.MAGE, value: 10, description: 'Attack +10' },
      { type: 'unlock', value: 'frost_nova', description: 'Unlock Frost Nova spell' },
      { type: 'unlock', value: 'heal', description: 'Unlock Heal spell' },
    ],
  },
  {
    id: 'arcane_mastery',
    name: 'Arcane Mastery',
    description: 'Mages +20 attack, unlock ultimate spells',
    category: ResearchCategory.MAGIC,
    cost: { gold: 3500, wood: 0, stone: 0, food: 1500 },
    researchTime: 240,
    prerequisites: ['advanced_magic'],
    effects: [
      { type: 'unit_stat', target: UnitType.MAGE, value: 20, description: 'Attack +20' },
      { type: 'unlock', value: 'meteor', description: 'Unlock Meteor spell' },
      { type: 'unlock', value: 'teleport', description: 'Unlock Teleport spell' },
    ],
  },
  {
    id: 'divine_blessing',
    name: 'Divine Blessing',
    description: 'All units regenerate 1 HP per second',
    category: ResearchCategory.MAGIC,
    cost: { gold: 1500, wood: 0, stone: 0, food: 1000 },
    researchTime: 150,
    prerequisites: ['basic_magic'],
    effects: [
      { type: 'special_ability', value: 'regeneration', description: 'All units regenerate HP' },
    ],
  },

  // Special Research
  {
    id: 'hero_training',
    name: 'Hero Training',
    description: 'Unlock hero units - powerful unique commanders',
    category: ResearchCategory.SPECIAL,
    cost: { gold: 2500, wood: 1000, stone: 1000, food: 1000 },
    researchTime: 200,
    prerequisites: [],
    effects: [
      { type: 'unlock', value: 'heroes', description: 'Unlock hero units' },
    ],
  },
  {
    id: 'legendary_heroes',
    name: 'Legendary Heroes',
    description: 'Heroes gain +50% stats and special abilities',
    category: ResearchCategory.SPECIAL,
    cost: { gold: 5000, wood: 2000, stone: 2000, food: 2000 },
    researchTime: 300,
    prerequisites: ['hero_training'],
    effects: [
      { type: 'unit_stat', target: 'heroes', value: 50, description: 'All hero stats +50%' },
      { type: 'unlock', value: 'hero_abilities', description: 'Unlock hero special abilities' },
    ],
  },
  {
    id: 'spy_network',
    name: 'Spy Network',
    description: 'Reveal enemy units and buildings on minimap',
    category: ResearchCategory.SPECIAL,
    cost: { gold: 1200, wood: 0, stone: 0, food: 800 },
    researchTime: 120,
    prerequisites: [],
    effects: [
      { type: 'special_ability', value: 'vision', description: 'Reveal enemy positions' },
    ],
  },
  {
    id: 'wonder',
    name: 'Ancient Wonder',
    description: 'Unlock the ability to build a Wonder - win condition',
    category: ResearchCategory.SPECIAL,
    cost: { gold: 10000, wood: 5000, stone: 5000, food: 5000 },
    researchTime: 600,
    prerequisites: ['architecture', 'legendary_heroes'],
    effects: [
      { type: 'unlock', value: 'wonder', description: 'Unlock Wonder construction' },
    ],
  },
];

// Research store state
interface ResearchState {
  completedResearch: Set<string>;
  activeResearch: ActiveResearch | null;
  queuedResearch: string[];
}

class ResearchStore {
  private state: ResearchState = {
    completedResearch: new Set(),
    activeResearch: null,
    queuedResearch: [],
  };

  private listeners: Set<(state: ResearchState) => void> = new Set();
  private updateInterval: NodeJS.Timeout | null = null;

  constructor() {
    // Update active research every second
    this.updateInterval = setInterval(() => {
      if (this.state.activeResearch) {
        const now = Date.now();
        if (now >= this.state.activeResearch.endTime) {
          this.completeResearch(this.state.activeResearch.researchId);
        }
      }
    }, 1000);
  }

  getState(): ResearchState {
    return this.state;
  }

  subscribe(listener: (state: ResearchState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this.state));
  }

  // Check if research is available
  canResearch(researchId: string): boolean {
    const research = RESEARCH_TREE.find(r => r.id === researchId);
    if (!research) return false;

    // Already completed
    if (this.state.completedResearch.has(researchId)) return false;

    // Already researching
    if (this.state.activeResearch?.researchId === researchId) return false;

    // Check prerequisites
    for (const prereq of research.prerequisites) {
      if (!this.state.completedResearch.has(prereq)) {
        return false;
      }
    }

    // Check resources
    if (!gameStore.canAfford(research.cost)) return false;

    return true;
  }

  // Start research
  startResearch(researchId: string): boolean {
    if (!this.canResearch(researchId)) return false;
    if (this.state.activeResearch) return false; // Already researching something

    const research = RESEARCH_TREE.find(r => r.id === researchId);
    if (!research) return false;

    // Spend resources
    if (!gameStore.spendResources(research.cost)) return false;

    // Start research
    const now = Date.now();
    this.state.activeResearch = {
      researchId,
      startTime: now,
      endTime: now + research.researchTime * 1000,
    };

    this.notifyListeners();
    return true;
  }

  // Complete research
  private completeResearch(researchId: string): void {
    const research = RESEARCH_TREE.find(r => r.id === researchId);
    if (!research) return;

    this.state.completedResearch.add(researchId);
    this.state.activeResearch = null;

    // Apply research effects
    this.applyResearchEffects(research);

    // Start next queued research
    if (this.state.queuedResearch.length > 0) {
      const nextResearch = this.state.queuedResearch.shift();
      if (nextResearch) {
        this.startResearch(nextResearch);
      }
    }

    this.notifyListeners();
  }

  // Apply research effects to game state
  private applyResearchEffects(research: Research): void {
    // Effects are tracked and applied by the game systems
    // This method can be expanded to directly modify game state
    console.log(`Research completed: ${research.name}`);
    research.effects.forEach(effect => {
      console.log(`  - ${effect.description}`);
    });
  }

  // Queue research
  queueResearch(researchId: string): boolean {
    if (this.state.queuedResearch.includes(researchId)) return false;
    this.state.queuedResearch.push(researchId);
    this.notifyListeners();
    return true;
  }

  // Cancel active research
  cancelResearch(): boolean {
    if (!this.state.activeResearch) return false;

    const research = RESEARCH_TREE.find(r => r.id === this.state.activeResearch!.researchId);
    if (research) {
      // Refund 50% of resources
      const refund: Resources = {
        gold: Math.floor(research.cost.gold * 0.5),
        wood: Math.floor(research.cost.wood * 0.5),
        stone: Math.floor(research.cost.stone * 0.5),
        food: Math.floor(research.cost.food * 0.5),
      };
      gameStore.addResources(refund);
    }

    this.state.activeResearch = null;
    this.notifyListeners();
    return true;
  }

  // Get research progress (0-1)
  getResearchProgress(): number {
    if (!this.state.activeResearch) return 0;

    const now = Date.now();
    const total = this.state.activeResearch.endTime - this.state.activeResearch.startTime;
    const elapsed = now - this.state.activeResearch.startTime;

    return Math.min(1, elapsed / total);
  }

  // Check if research is completed
  hasResearch(researchId: string): boolean {
    return this.state.completedResearch.has(researchId);
  }

  // Get available research
  getAvailableResearch(): Research[] {
    return RESEARCH_TREE.filter(r => this.canResearch(r.id));
  }

  // Get completed research
  getCompletedResearch(): Research[] {
    return RESEARCH_TREE.filter(r => this.state.completedResearch.has(r.id));
  }

  // Cleanup
  destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
}

export const researchStore = new ResearchStore();
