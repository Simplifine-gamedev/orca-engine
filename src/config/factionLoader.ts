// Faction loader for dynamically loading faction-specific unit configs
// This module loads JSON configurations from generated_factions directory

import { UnitConfig, FactionConfig } from './factions';
import * as fs from 'fs';
import * as path from 'path';

export interface ExtendedUnitConfig extends UnitConfig {
  faction?: string;
  specialAbilities?: Array<{
    name: string;
    description: string;
    cooldown: number;
    duration?: number;
    passive?: boolean;
    damage?: number;
    radius?: number;
  }>;
  bonuses?: Record<string, any>;
  voiceLines?: {
    onCreated?: string;
    onSelected?: string;
    onMove?: string;
    onAttack?: string;
  };
}

export class FactionLoader {
  private loadedFactions: Map<string, FactionConfig> = new Map();
  private loadedUnits: Map<string, ExtendedUnitConfig> = new Map();

  constructor(private factionsDir: string = './generated_factions') {}

  // Load all faction configurations from JSON files
  loadAllFactions(): void {
    try {
      const files = fs.readdirSync(this.factionsDir);
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          const filePath = path.join(this.factionsDir, file);
          this.loadUnitFromFile(filePath);
        }
      }

      console.log(`Loaded ${this.loadedUnits.size} faction-specific units`);
    } catch (error) {
      console.error('Error loading factions:', error);
    }
  }

  // Load a single unit configuration from a JSON file
  private loadUnitFromFile(filePath: string): void {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const unitConfig: ExtendedUnitConfig = JSON.parse(content);
      
      this.loadedUnits.set(unitConfig.id, unitConfig);
      
      // Group by faction
      const factionId = unitConfig.faction || 'unknown';
      if (!this.loadedFactions.has(factionId)) {
        this.loadedFactions.set(factionId, {
          id: factionId,
          name: this.capitalizeFirst(factionId),
          description: `${this.capitalizeFirst(factionId)} faction`,
          units: [],
        });
      }
      
      const faction = this.loadedFactions.get(factionId)!;
      faction.units.push(unitConfig);
    } catch (error) {
      console.error(`Error loading unit from ${filePath}:`, error);
    }
  }

  // Get all loaded factions
  getFactions(): FactionConfig[] {
    return Array.from(this.loadedFactions.values());
  }

  // Get faction by id
  getFaction(factionId: string): FactionConfig | undefined {
    return this.loadedFactions.get(factionId);
  }

  // Get unit by id
  getUnit(unitId: string): ExtendedUnitConfig | undefined {
    return this.loadedUnits.get(unitId);
  }

  // Get all scout units
  getScoutUnits(): ExtendedUnitConfig[] {
    return Array.from(this.loadedUnits.values()).filter(u => u.type === 'scout');
  }

  // Get units by faction
  getUnitsByFaction(factionId: string): ExtendedUnitConfig[] {
    return Array.from(this.loadedUnits.values()).filter(u => u.faction === factionId);
  }

  // Get units available from a specific building
  getUnitsFromBuilding(buildingType: string): ExtendedUnitConfig[] {
    return Array.from(this.loadedUnits.values()).filter(
      u => u.availableFrom.includes(buildingType)
    );
  }

  // Helper to capitalize first letter
  private capitalizeFirst(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  // Export all units as a single JSON file
  exportAllUnits(outputPath: string): void {
    const units = Array.from(this.loadedUnits.values());
    fs.writeFileSync(outputPath, JSON.stringify(units, null, 2));
    console.log(`Exported ${units.length} units to ${outputPath}`);
  }

  // Get unit statistics summary
  getStatsSummary(): Record<string, any> {
    const scouts = this.getScoutUnits();
    
    return {
      totalUnits: this.loadedUnits.size,
      totalFactions: this.loadedFactions.size,
      scoutUnits: scouts.length,
      averageScoutSpeed: scouts.reduce((sum, u) => sum + u.stats.movementSpeed, 0) / scouts.length,
      averageScoutVision: scouts.reduce((sum, u) => sum + u.stats.visionRange, 0) / scouts.length,
      averageScoutCost: {
        gold: scouts.reduce((sum, u) => sum + (u.cost.gold || 0), 0) / scouts.length,
        food: scouts.reduce((sum, u) => sum + (u.cost.food || 0), 0) / scouts.length,
      },
    };
  }
}

// Singleton instance
let factionLoaderInstance: FactionLoader | null = null;

export function getFactionLoader(factionsDir?: string): FactionLoader {
  if (!factionLoaderInstance) {
    factionLoaderInstance = new FactionLoader(factionsDir);
    factionLoaderInstance.loadAllFactions();
  }
  return factionLoaderInstance;
}

export default FactionLoader;
