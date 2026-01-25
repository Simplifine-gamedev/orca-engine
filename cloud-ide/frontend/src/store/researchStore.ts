// Research system state management
import { ResearchTech, ResearchProgress, GameResources } from '../types/game';
import { RESEARCH_TECHS } from '../data/researchTechs';

export interface ResearchState {
  completedResearch: string[];
  activeResearch: ResearchProgress | null;
  availableTechs: ResearchTech[];
}

class ResearchStore {
  private state: ResearchState = {
    completedResearch: [],
    activeResearch: null,
    availableTechs: Object.values(RESEARCH_TECHS),
  };
  
  private listeners: Set<() => void> = new Set();

  /**
   * Subscribe to state changes
   */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify all subscribers of state change
   */
  private notify(): void {
    this.listeners.forEach(listener => listener());
  }

  /**
   * Get current research state
   */
  getState(): ResearchState {
    return { ...this.state };
  }

  /**
   * Check if a technology is completed
   */
  isResearchCompleted(techId: string): boolean {
    return this.state.completedResearch.includes(techId);
  }

  /**
   * Check if prerequisites are met for a research
   */
  canStartResearch(techId: string, playerResources: GameResources): boolean {
    const tech = RESEARCH_TECHS[techId];
    if (!tech) return false;

    // Check if already researched
    if (this.isResearchCompleted(techId)) return false;

    // Check if already researching something
    if (this.state.activeResearch) return false;

    // Check prerequisites
    const hasPrerequisites = tech.prerequisites.every(prereq =>
      this.isResearchCompleted(prereq)
    );
    if (!hasPrerequisites) return false;

    // Check resources
    if (tech.cost.gold && playerResources.gold < tech.cost.gold) return false;
    if (tech.cost.wood && playerResources.wood < tech.cost.wood) return false;
    if (tech.cost.stone && playerResources.stone < tech.cost.stone) return false;
    if (tech.cost.food && playerResources.food < tech.cost.food) return false;

    return true;
  }

  /**
   * Start researching a technology
   */
  startResearch(techId: string, playerResources: GameResources): boolean {
    if (!this.canStartResearch(techId, playerResources)) {
      return false;
    }

    const tech = RESEARCH_TECHS[techId];
    if (!tech) return false;

    this.state.activeResearch = {
      techId,
      progress: 0,
      isComplete: false,
      startTime: Date.now(),
    };

    this.notify();
    return true;
  }

  /**
   * Cancel active research
   */
  cancelResearch(): void {
    this.state.activeResearch = null;
    this.notify();
  }

  /**
   * Update research progress (call this periodically)
   */
  updateResearch(deltaTime: number): void {
    if (!this.state.activeResearch) return;

    const tech = RESEARCH_TECHS[this.state.activeResearch.techId];
    if (!tech) return;

    // Update progress
    this.state.activeResearch.progress += deltaTime;

    // Check if complete
    if (this.state.activeResearch.progress >= tech.researchTime) {
      this.completeResearch(this.state.activeResearch.techId);
    } else {
      this.notify();
    }
  }

  /**
   * Complete a research
   */
  private completeResearch(techId: string): void {
    this.state.completedResearch.push(techId);
    this.state.activeResearch = null;
    this.notify();
  }

  /**
   * Get active research progress as percentage
   */
  getResearchProgress(): number {
    if (!this.state.activeResearch) return 0;

    const tech = RESEARCH_TECHS[this.state.activeResearch.techId];
    if (!tech) return 0;

    return Math.min(100, (this.state.activeResearch.progress / tech.researchTime) * 100);
  }

  /**
   * Get available technologies for a specific building
   */
  getTechsForBuilding(buildingId: string): ResearchTech[] {
    return this.state.availableTechs.filter(
      tech => tech.buildingRequired === buildingId
    );
  }

  /**
   * Reset research state (for testing/debugging)
   */
  reset(): void {
    this.state = {
      completedResearch: [],
      activeResearch: null,
      availableTechs: Object.values(RESEARCH_TECHS),
    };
    this.notify();
  }
}

// Export singleton instance
export const researchStore = new ResearchStore();
