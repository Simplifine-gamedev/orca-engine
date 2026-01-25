import {
  Research,
  ResearchCategory,
  ResearchStatus,
  ResearchProgress,
  EffectType,
  ResearchCost,
} from '../types/research';

// Research Store - Manages research tree and progression
export class ResearchStore {
  private researches: Map<string, Research> = new Map();
  private completedResearch: Set<string> = new Set();
  private currentResearch: ResearchProgress | null = null;
  private researchQueue: string[] = [];
  
  constructor() {
    this.initializeResearchTree();
  }

  // Initialize the complete research tree with all upgrades
  private initializeResearchTree() {
    const researchData: Research[] = [
      // MILITARY RESEARCH
      {
        id: 'melee_attack_1',
        name: 'Forged Blades',
        description: 'Improve melee weapon quality, increasing attack damage by 10%',
        category: ResearchCategory.MILITARY,
        cost: { gold: 200, wood: 100 },
        researchTime: 30,
        prerequisites: [],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'warrior',
            value: 1.1,
            description: '+10% attack damage for warriors',
          },
        ],
      },
      {
        id: 'melee_attack_2',
        name: 'Steel Weapons',
        description: 'Advanced metallurgy increases melee attack damage by 20%',
        category: ResearchCategory.MILITARY,
        cost: { gold: 400, wood: 200, stone: 100 },
        researchTime: 60,
        prerequisites: ['melee_attack_1'],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'warrior',
            value: 1.2,
            description: '+20% attack damage for warriors',
          },
        ],
      },
      {
        id: 'melee_attack_3',
        name: 'Enchanted Weapons',
        description: 'Magic-infused weapons grant 35% increased attack damage',
        category: ResearchCategory.MILITARY,
        cost: { gold: 800, mana: 200, stone: 200 },
        researchTime: 120,
        prerequisites: ['melee_attack_2', 'basic_magic'],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'warrior',
            value: 1.35,
            description: '+35% attack damage for warriors',
          },
        ],
      },
      {
        id: 'armor_1',
        name: 'Leather Armor',
        description: 'Basic armor training increases unit armor by 1',
        category: ResearchCategory.MILITARY,
        cost: { gold: 150, wood: 50 },
        researchTime: 25,
        prerequisites: [],
        effects: [
          {
            type: EffectType.ARMOR,
            target: 'warrior',
            value: 1,
            description: '+1 armor for warriors',
          },
        ],
      },
      {
        id: 'armor_2',
        name: 'Chainmail Armor',
        description: 'Metal armor provides +2 armor',
        category: ResearchCategory.MILITARY,
        cost: { gold: 300, stone: 150 },
        researchTime: 50,
        prerequisites: ['armor_1'],
        effects: [
          {
            type: EffectType.ARMOR,
            target: 'warrior',
            value: 2,
            description: '+2 armor for warriors',
          },
        ],
      },
      {
        id: 'armor_3',
        name: 'Plate Armor',
        description: 'Heavy plate armor grants +4 armor',
        category: ResearchCategory.MILITARY,
        cost: { gold: 600, stone: 300 },
        researchTime: 100,
        prerequisites: ['armor_2'],
        effects: [
          {
            type: EffectType.ARMOR,
            target: 'warrior',
            value: 4,
            description: '+4 armor for warriors',
          },
        ],
      },
      {
        id: 'ranged_attack_1',
        name: 'Fletcher',
        description: 'Better arrows increase ranged attack by 15%',
        category: ResearchCategory.MILITARY,
        cost: { gold: 180, wood: 120 },
        researchTime: 30,
        prerequisites: [],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'archer',
            value: 1.15,
            description: '+15% attack damage for archers',
          },
        ],
      },
      {
        id: 'ranged_attack_2',
        name: 'Bodkin Arrows',
        description: 'Armor-piercing arrows grant +25% ranged damage',
        category: ResearchCategory.MILITARY,
        cost: { gold: 350, wood: 250, stone: 50 },
        researchTime: 60,
        prerequisites: ['ranged_attack_1'],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'archer',
            value: 1.25,
            description: '+25% attack damage for archers',
          },
        ],
      },
      {
        id: 'cavalry_speed',
        name: 'Horseback Riding',
        description: 'Cavalry moves 20% faster',
        category: ResearchCategory.MILITARY,
        cost: { gold: 250, food: 150 },
        researchTime: 40,
        prerequisites: [],
        effects: [
          {
            type: EffectType.MOVEMENT_SPEED,
            target: 'cavalry',
            value: 1.2,
            description: '+20% movement speed for cavalry',
          },
        ],
      },
      {
        id: 'hero_unit',
        name: 'Hero Training',
        description: 'Unlock powerful hero units',
        category: ResearchCategory.MILITARY,
        cost: { gold: 1000, mana: 300 },
        researchTime: 180,
        prerequisites: ['melee_attack_2', 'armor_2'],
        effects: [
          {
            type: EffectType.UNLOCK_UNIT,
            target: 'hero',
            value: 1,
            description: 'Unlock hero units',
          },
        ],
      },

      // ECONOMY RESEARCH
      {
        id: 'wood_gathering_1',
        name: 'Double-Bit Axe',
        description: 'Better axes increase wood gathering by 15%',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 100, wood: 50 },
        researchTime: 20,
        prerequisites: [],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'wood',
            value: 1.15,
            description: '+15% wood gathering speed',
          },
        ],
      },
      {
        id: 'wood_gathering_2',
        name: 'Bow Saw',
        description: 'Advanced tools grant +30% wood gathering',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 200, wood: 100, stone: 50 },
        researchTime: 40,
        prerequisites: ['wood_gathering_1'],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'wood',
            value: 1.3,
            description: '+30% wood gathering speed',
          },
        ],
      },
      {
        id: 'gold_mining_1',
        name: 'Gold Mining',
        description: 'Better pickaxes increase gold gathering by 15%',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 120, stone: 80 },
        researchTime: 25,
        prerequisites: [],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'gold',
            value: 1.15,
            description: '+15% gold gathering speed',
          },
        ],
      },
      {
        id: 'gold_mining_2',
        name: 'Gold Shaft Mining',
        description: 'Deep mining techniques grant +30% gold gathering',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 250, stone: 150 },
        researchTime: 50,
        prerequisites: ['gold_mining_1'],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'gold',
            value: 1.3,
            description: '+30% gold gathering speed',
          },
        ],
      },
      {
        id: 'stone_mining_1',
        name: 'Stone Mining',
        description: 'Stone gathering improved by 15%',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 100, wood: 80 },
        researchTime: 20,
        prerequisites: [],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'stone',
            value: 1.15,
            description: '+15% stone gathering speed',
          },
        ],
      },
      {
        id: 'farming_1',
        name: 'Crop Rotation',
        description: 'Better farming increases food production by 20%',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 150, wood: 100 },
        researchTime: 30,
        prerequisites: [],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'food',
            value: 1.2,
            description: '+20% food production',
          },
        ],
      },
      {
        id: 'farming_2',
        name: 'Irrigation',
        description: 'Advanced farming grants +40% food production',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 300, wood: 200, stone: 100 },
        researchTime: 60,
        prerequisites: ['farming_1'],
        effects: [
          {
            type: EffectType.GATHERING_SPEED,
            target: 'food',
            value: 1.4,
            description: '+40% food production',
          },
        ],
      },
      {
        id: 'wheelbarrow',
        name: 'Wheelbarrow',
        description: 'Workers carry 25% more resources',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 200, wood: 150 },
        researchTime: 35,
        prerequisites: [],
        effects: [
          {
            type: EffectType.RESOURCE_CAPACITY,
            target: 'worker',
            value: 1.25,
            description: '+25% carrying capacity',
          },
        ],
      },
      {
        id: 'hand_cart',
        name: 'Hand Cart',
        description: 'Workers carry 50% more resources',
        category: ResearchCategory.ECONOMY,
        cost: { gold: 400, wood: 300, stone: 100 },
        researchTime: 70,
        prerequisites: ['wheelbarrow'],
        effects: [
          {
            type: EffectType.RESOURCE_CAPACITY,
            target: 'worker',
            value: 1.5,
            description: '+50% carrying capacity',
          },
        ],
      },

      // TECHNOLOGY RESEARCH
      {
        id: 'masonry',
        name: 'Masonry',
        description: 'Buildings have +10% HP and cost 5% less stone',
        category: ResearchCategory.TECHNOLOGY,
        cost: { gold: 300, stone: 200 },
        researchTime: 45,
        prerequisites: [],
        effects: [
          {
            type: EffectType.HP,
            target: 'buildings',
            value: 1.1,
            description: '+10% building HP',
          },
          {
            type: EffectType.COST_REDUCTION,
            target: 'stone',
            value: 0.95,
            description: '-5% stone cost for buildings',
          },
        ],
      },
      {
        id: 'architecture',
        name: 'Architecture',
        description: 'Buildings have +20% HP and build 15% faster',
        category: ResearchCategory.TECHNOLOGY,
        cost: { gold: 600, stone: 400, wood: 200 },
        researchTime: 90,
        prerequisites: ['masonry'],
        effects: [
          {
            type: EffectType.HP,
            target: 'buildings',
            value: 1.2,
            description: '+20% building HP',
          },
          {
            type: EffectType.BUILDING_SPEED,
            target: 'global',
            value: 1.15,
            description: '+15% building construction speed',
          },
        ],
      },
      {
        id: 'advanced_architecture',
        name: 'Advanced Architecture',
        description: 'Unlock advanced buildings and +30% building HP',
        category: ResearchCategory.TECHNOLOGY,
        cost: { gold: 1200, stone: 800, wood: 400 },
        researchTime: 150,
        prerequisites: ['architecture'],
        effects: [
          {
            type: EffectType.HP,
            target: 'buildings',
            value: 1.3,
            description: '+30% building HP',
          },
          {
            type: EffectType.UNLOCK_BUILDING,
            target: 'academy',
            value: 1,
            description: 'Unlock Academy',
          },
          {
            type: EffectType.UNLOCK_BUILDING,
            target: 'workshop',
            value: 1,
            description: 'Unlock Workshop',
          },
        ],
      },
      {
        id: 'ballistics',
        name: 'Ballistics',
        description: 'Unlock siege units',
        category: ResearchCategory.TECHNOLOGY,
        cost: { gold: 500, wood: 300, stone: 200 },
        researchTime: 80,
        prerequisites: ['masonry'],
        effects: [
          {
            type: EffectType.UNLOCK_UNIT,
            target: 'siege',
            value: 1,
            description: 'Unlock siege units',
          },
        ],
      },
      {
        id: 'conscription',
        name: 'Conscription',
        description: 'Units train 30% faster',
        category: ResearchCategory.TECHNOLOGY,
        cost: { gold: 400, food: 300 },
        researchTime: 60,
        prerequisites: [],
        effects: [
          {
            type: EffectType.TRAINING_TIME,
            target: 'global',
            value: 0.7,
            description: '-30% unit training time',
          },
        ],
      },

      // MAGIC RESEARCH
      {
        id: 'basic_magic',
        name: 'Basic Magic',
        description: 'Unlock mage units and basic spells',
        category: ResearchCategory.MAGIC,
        cost: { gold: 300, mana: 100 },
        researchTime: 50,
        prerequisites: [],
        effects: [
          {
            type: EffectType.UNLOCK_UNIT,
            target: 'mage',
            value: 1,
            description: 'Unlock mage units',
          },
        ],
      },
      {
        id: 'advanced_magic',
        name: 'Advanced Magic',
        description: 'Mages deal 25% more damage',
        category: ResearchCategory.MAGIC,
        cost: { gold: 600, mana: 300 },
        researchTime: 90,
        prerequisites: ['basic_magic'],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'mage',
            value: 1.25,
            description: '+25% spell damage',
          },
        ],
      },
      {
        id: 'arcane_mastery',
        name: 'Arcane Mastery',
        description: 'Unlock powerful spells and increase mage damage by 50%',
        category: ResearchCategory.MAGIC,
        cost: { gold: 1200, mana: 600 },
        researchTime: 180,
        prerequisites: ['advanced_magic'],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'mage',
            value: 1.5,
            description: '+50% spell damage',
          },
          {
            type: EffectType.UNLOCK_ABILITY,
            target: 'meteor',
            value: 1,
            description: 'Unlock Meteor spell',
          },
        ],
      },
      {
        id: 'healing',
        name: 'Healing Magic',
        description: 'Unlock priest units with healing abilities',
        category: ResearchCategory.MAGIC,
        cost: { gold: 400, mana: 200 },
        researchTime: 60,
        prerequisites: ['basic_magic'],
        effects: [
          {
            type: EffectType.UNLOCK_UNIT,
            target: 'priest',
            value: 1,
            description: 'Unlock priest units',
          },
        ],
      },

      // DEFENSE RESEARCH
      {
        id: 'fortification_1',
        name: 'Fortification',
        description: 'Unlock towers and walls',
        category: ResearchCategory.DEFENSE,
        cost: { gold: 200, stone: 150 },
        researchTime: 40,
        prerequisites: [],
        effects: [
          {
            type: EffectType.UNLOCK_BUILDING,
            target: 'tower',
            value: 1,
            description: 'Unlock towers',
          },
          {
            type: EffectType.UNLOCK_BUILDING,
            target: 'wall',
            value: 1,
            description: 'Unlock walls',
          },
        ],
      },
      {
        id: 'fortification_2',
        name: 'Advanced Fortification',
        description: 'Towers have +50% range and damage',
        category: ResearchCategory.DEFENSE,
        cost: { gold: 400, stone: 300 },
        researchTime: 80,
        prerequisites: ['fortification_1'],
        effects: [
          {
            type: EffectType.ATTACK_DAMAGE,
            target: 'tower',
            value: 1.5,
            description: '+50% tower damage',
          },
          {
            type: EffectType.VISION_RANGE,
            target: 'tower',
            value: 1.5,
            description: '+50% tower range',
          },
        ],
      },
      {
        id: 'guard_towers',
        name: 'Guard Towers',
        description: 'Towers can garrison units',
        category: ResearchCategory.DEFENSE,
        cost: { gold: 600, stone: 400 },
        researchTime: 100,
        prerequisites: ['fortification_2'],
        effects: [
          {
            type: EffectType.UNLOCK_ABILITY,
            target: 'garrison',
            value: 1,
            description: 'Towers can garrison units',
          },
        ],
      },
    ];

    researchData.forEach((research) => {
      this.researches.set(research.id, research);
    });
  }

  // Get research status
  getResearchStatus(researchId: string): ResearchStatus {
    if (this.completedResearch.has(researchId)) {
      return ResearchStatus.COMPLETED;
    }
    
    if (this.currentResearch && this.currentResearch.researchId === researchId) {
      return ResearchStatus.RESEARCHING;
    }
    
    const research = this.researches.get(researchId);
    if (!research) {
      return ResearchStatus.LOCKED;
    }
    
    // Check prerequisites
    const prerequisitesMet = research.prerequisites.every((prereqId) =>
      this.completedResearch.has(prereqId)
    );
    
    return prerequisitesMet ? ResearchStatus.AVAILABLE : ResearchStatus.LOCKED;
  }

  // Get all researches by category
  getResearchesByCategory(category: ResearchCategory): Research[] {
    return Array.from(this.researches.values()).filter(
      (r) => r.category === category
    );
  }

  // Get all available researches
  getAvailableResearches(): Research[] {
    return Array.from(this.researches.values()).filter(
      (r) => this.getResearchStatus(r.id) === ResearchStatus.AVAILABLE
    );
  }

  // Start research
  startResearch(researchId: string, resources: ResourceCheck): boolean {
    const research = this.researches.get(researchId);
    if (!research) {
      return false;
    }

    const status = this.getResearchStatus(researchId);
    if (status !== ResearchStatus.AVAILABLE) {
      return false;
    }

    // Check if we have enough resources
    if (!this.canAfford(research.cost, resources)) {
      return false;
    }

    // Start research
    this.currentResearch = {
      researchId,
      startTime: Date.now(),
      totalTime: research.researchTime * 1000, // convert to ms
      progress: 0,
    };

    return true;
  }

  // Update research progress
  updateResearch(deltaTime: number): ResearchProgress | null {
    if (!this.currentResearch) {
      return null;
    }

    const elapsed = Date.now() - this.currentResearch.startTime;
    this.currentResearch.progress = Math.min(
      1,
      elapsed / this.currentResearch.totalTime
    );

    // Check if completed
    if (this.currentResearch.progress >= 1) {
      const completedId = this.currentResearch.researchId;
      this.completedResearch.add(completedId);
      this.currentResearch = null;

      // Start next in queue if available
      if (this.researchQueue.length > 0) {
        const nextId = this.researchQueue.shift()!;
        // Note: This would need resource check from game store
        // For now, just clear the queue item
      }

      return {
        researchId: completedId,
        startTime: 0,
        totalTime: 0,
        progress: 1,
      };
    }

    return this.currentResearch;
  }

  // Queue research
  queueResearch(researchId: string): boolean {
    if (this.researchQueue.includes(researchId)) {
      return false;
    }
    
    this.researchQueue.push(researchId);
    return true;
  }

  // Cancel current research
  cancelResearch(): boolean {
    if (!this.currentResearch) {
      return false;
    }

    // Could implement partial refund here
    this.currentResearch = null;
    return true;
  }

  // Get completed researches
  getCompletedResearches(): Set<string> {
    return new Set(this.completedResearch);
  }

  // Check if can afford research
  private canAfford(cost: ResearchCost, resources: ResourceCheck): boolean {
    if (resources.gold < cost.gold) return false;
    if (cost.wood && resources.wood < cost.wood) return false;
    if (cost.stone && resources.stone < cost.stone) return false;
    if (cost.food && resources.food < cost.food) return false;
    if (cost.mana && resources.mana < cost.mana) return false;
    return true;
  }

  // Get current research progress
  getCurrentResearch(): ResearchProgress | null {
    return this.currentResearch;
  }

  // Get research by ID
  getResearch(id: string): Research | undefined {
    return this.researches.get(id);
  }

  // Get all researches
  getAllResearches(): Research[] {
    return Array.from(this.researches.values());
  }
}

// Helper interface for resource checking
interface ResourceCheck {
  gold: number;
  wood: number;
  stone: number;
  food: number;
  mana: number;
}
