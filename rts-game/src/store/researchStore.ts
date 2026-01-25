import { create } from 'zustand';
import { ResearchTech, ResearchProgress } from '../types/research';
import { blacksmithResearch, canResearch } from '../buildings/buildingModels';

interface ResearchState {
  completedResearch: ResearchTech[];
  activeResearch: ResearchProgress | null;
  availableGold: number;
  availableFood: number;
  
  // Actions
  startResearch: (techId: ResearchTech) => boolean;
  completeResearch: (techId: ResearchTech) => void;
  cancelResearch: () => void;
  updateResearchProgress: (progress: number) => void;
  setResources: (gold: number, food: number) => void;
  addResources: (gold: number, food: number) => void;
}

export const useResearchStore = create<ResearchState>((set, get) => ({
  completedResearch: [],
  activeResearch: null,
  availableGold: 1000,
  availableFood: 500,
  
  startResearch: (techId: ResearchTech) => {
    const state = get();
    
    // Check if research is already active
    if (state.activeResearch) {
      console.warn('Research already in progress');
      return false;
    }
    
    // Check if research is already completed
    if (state.completedResearch.includes(techId)) {
      console.warn('Research already completed');
      return false;
    }
    
    // Check prerequisites
    if (!canResearch(techId, state.completedResearch)) {
      console.warn('Prerequisites not met');
      return false;
    }
    
    // Check resources
    const tech = blacksmithResearch[techId];
    if (state.availableGold < tech.cost.gold || 
        (tech.cost.food && state.availableFood < tech.cost.food)) {
      console.warn('Insufficient resources');
      return false;
    }
    
    // Deduct resources and start research
    set({
      availableGold: state.availableGold - tech.cost.gold,
      availableFood: state.availableFood - (tech.cost.food || 0),
      activeResearch: {
        techId,
        progress: 0,
        startTime: Date.now(),
      },
    });
    
    return true;
  },
  
  completeResearch: (techId: ResearchTech) => {
    set((state) => ({
      completedResearch: [...state.completedResearch, techId],
      activeResearch: null,
    }));
  },
  
  cancelResearch: () => {
    const state = get();
    if (state.activeResearch) {
      // Refund partial resources based on progress
      const tech = blacksmithResearch[state.activeResearch.techId];
      const refundPercent = Math.max(0, 1 - state.activeResearch.progress / 100) * 0.5; // 50% max refund
      
      set({
        availableGold: state.availableGold + Math.floor(tech.cost.gold * refundPercent),
        availableFood: state.availableFood + Math.floor((tech.cost.food || 0) * refundPercent),
        activeResearch: null,
      });
    }
  },
  
  updateResearchProgress: (progress: number) => {
    set((state) => ({
      activeResearch: state.activeResearch 
        ? { ...state.activeResearch, progress }
        : null,
    }));
  },
  
  setResources: (gold: number, food: number) => {
    set({ availableGold: gold, availableFood: food });
  },
  
  addResources: (gold: number, food: number) => {
    set((state) => ({
      availableGold: state.availableGold + gold,
      availableFood: state.availableFood + food,
    }));
  },
}));

// Auto-progress research
if (typeof window !== 'undefined') {
  setInterval(() => {
    const state = useResearchStore.getState();
    if (state.activeResearch) {
      const tech = blacksmithResearch[state.activeResearch.techId];
      const elapsed = (Date.now() - state.activeResearch.startTime) / 1000;
      const progress = Math.min(100, (elapsed / tech.researchTime) * 100);
      
      state.updateResearchProgress(progress);
      
      if (progress >= 100) {
        state.completeResearch(state.activeResearch.techId);
      }
    }
  }, 100); // Update every 100ms
}
