/**
 * Research Panel UI Component
 * Displays the research tree and allows players to start/queue research
 */

import React, { useState, useEffect } from 'react';
import { 
  researchStore, 
  RESEARCH_TREE, 
  Research, 
  ResearchCategory 
} from '../store/researchStore';
import { gameStore, Resources } from '../store/gameStore';

interface ResearchPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ResearchPanel: React.FC<ResearchPanelProps> = ({ isOpen, onClose }) => {
  const [selectedCategory, setSelectedCategory] = useState<ResearchCategory | 'all'>('all');
  const [selectedResearch, setSelectedResearch] = useState<Research | null>(null);
  const [gameState, setGameState] = useState(gameStore.getState());
  const [researchState, setResearchState] = useState(researchStore.getState());
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Subscribe to store updates
    const unsubGame = gameStore.subscribe(setGameState);
    const unsubResearch = researchStore.subscribe(setResearchState);

    // Update progress every 100ms when researching
    const progressInterval = setInterval(() => {
      if (researchState.activeResearch) {
        setProgress(researchStore.getResearchProgress());
      }
    }, 100);

    return () => {
      unsubGame();
      unsubResearch();
      clearInterval(progressInterval);
    };
  }, [researchState.activeResearch]);

  if (!isOpen) return null;

  const filteredResearch = RESEARCH_TREE.filter(
    r => selectedCategory === 'all' || r.category === selectedCategory
  );

  const activeResearchData = researchState.activeResearch
    ? RESEARCH_TREE.find(r => r.id === researchState.activeResearch!.researchId)
    : null;

  const canAffordResearch = (research: Research): boolean => {
    return gameStore.canAfford(research.cost);
  };

  const getResearchStatus = (research: Research): 'completed' | 'available' | 'locked' | 'active' => {
    if (researchState.completedResearch.has(research.id)) return 'completed';
    if (researchState.activeResearch?.researchId === research.id) return 'active';
    if (researchStore.canResearch(research.id)) return 'available';
    return 'locked';
  };

  const handleStartResearch = (researchId: string) => {
    if (researchStore.startResearch(researchId)) {
      console.log('Started research:', researchId);
    }
  };

  const handleCancelResearch = () => {
    if (researchStore.cancelResearch()) {
      setProgress(0);
    }
  };

  const formatResources = (resources: Resources): string => {
    const parts = [];
    if (resources.gold > 0) parts.push(`${resources.gold} gold`);
    if (resources.wood > 0) parts.push(`${resources.wood} wood`);
    if (resources.stone > 0) parts.push(`${resources.stone} stone`);
    if (resources.food > 0) parts.push(`${resources.food} food`);
    return parts.join(', ');
  };

  const getCategoryColor = (category: ResearchCategory): string => {
    switch (category) {
      case ResearchCategory.ECONOMY: return 'text-yellow-400';
      case ResearchCategory.MILITARY: return 'text-red-400';
      case ResearchCategory.TECHNOLOGY: return 'text-blue-400';
      case ResearchCategory.MAGIC: return 'text-purple-400';
      case ResearchCategory.SPECIAL: return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getCategoryBg = (category: ResearchCategory): string => {
    switch (category) {
      case ResearchCategory.ECONOMY: return 'bg-yellow-900/30 border-yellow-600';
      case ResearchCategory.MILITARY: return 'bg-red-900/30 border-red-600';
      case ResearchCategory.TECHNOLOGY: return 'bg-blue-900/30 border-blue-600';
      case ResearchCategory.MAGIC: return 'bg-purple-900/30 border-purple-600';
      case ResearchCategory.SPECIAL: return 'bg-green-900/30 border-green-600';
      default: return 'bg-gray-900/30 border-gray-600';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border-2 border-gray-700 rounded-lg w-5/6 h-5/6 flex flex-col shadow-2xl">
        {/* Header */}
        <div className="bg-gray-800 border-b-2 border-gray-700 p-4 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">Research & Technology</h2>
            <p className="text-sm text-gray-400">Advance your civilization through research</p>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-white font-semibold"
          >
            Close
          </button>
        </div>

        {/* Active Research Bar */}
        {activeResearchData && (
          <div className="bg-gray-800 border-b border-gray-700 p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className={`px-2 py-1 rounded text-xs font-semibold ${getCategoryColor(activeResearchData.category)}`}>
                  {activeResearchData.category.toUpperCase()}
                </div>
                <span className="text-white font-semibold">{activeResearchData.name}</span>
                <span className="text-gray-400 text-sm">({Math.round(progress * 100)}%)</span>
              </div>
              <button
                onClick={handleCancelResearch}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm text-white"
              >
                Cancel (50% refund)
              </button>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="bg-blue-500 h-3 rounded-full transition-all duration-100"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Resource Display */}
        <div className="bg-gray-800 border-b border-gray-700 p-3 flex gap-6">
          <div className="flex items-center gap-2">
            <span className="text-yellow-500 font-bold">Gold:</span>
            <span className="text-white">{gameState.resources.gold}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-yellow-700 font-bold">Wood:</span>
            <span className="text-white">{gameState.resources.wood}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-400 font-bold">Stone:</span>
            <span className="text-white">{gameState.resources.stone}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-green-500 font-bold">Food:</span>
            <span className="text-white">{gameState.resources.food}</span>
          </div>
        </div>

        {/* Category Filters */}
        <div className="bg-gray-800 border-b border-gray-700 p-3 flex gap-2">
          <button
            onClick={() => setSelectedCategory('all')}
            className={`px-4 py-2 rounded font-semibold ${
              selectedCategory === 'all'
                ? 'bg-gray-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
            }`}
          >
            All Research
          </button>
          {Object.values(ResearchCategory).map(category => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded font-semibold capitalize ${
                selectedCategory === category
                  ? `${getCategoryColor(category)} bg-gray-600`
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-650'
              }`}
            >
              {category}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Research List */}
          <div className="w-2/3 overflow-y-auto p-4">
            <div className="grid grid-cols-2 gap-4">
              {filteredResearch.map(research => {
                const status = getResearchStatus(research);
                const affordable = canAffordResearch(research);

                return (
                  <div
                    key={research.id}
                    onClick={() => setSelectedResearch(research)}
                    className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                      selectedResearch?.id === research.id
                        ? 'ring-2 ring-blue-500'
                        : ''
                    } ${getCategoryBg(research.category)} ${
                      status === 'completed'
                        ? 'opacity-60'
                        : status === 'locked'
                        ? 'opacity-40'
                        : 'hover:scale-105'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <div className={`text-sm font-semibold mb-1 ${getCategoryColor(research.category)}`}>
                          {research.category.toUpperCase()}
                        </div>
                        <h3 className="text-lg font-bold text-white">{research.name}</h3>
                      </div>
                      {status === 'completed' && (
                        <div className="bg-green-600 text-white px-2 py-1 rounded text-xs font-bold">
                          DONE
                        </div>
                      )}
                      {status === 'active' && (
                        <div className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-bold animate-pulse">
                          ACTIVE
                        </div>
                      )}
                      {status === 'locked' && (
                        <div className="bg-red-600 text-white px-2 py-1 rounded text-xs font-bold">
                          LOCKED
                        </div>
                      )}
                    </div>

                    <p className="text-sm text-gray-300 mb-3">{research.description}</p>

                    <div className="space-y-1 mb-3">
                      <div className="text-xs text-gray-400">
                        Cost: {formatResources(research.cost)}
                      </div>
                      <div className="text-xs text-gray-400">
                        Research Time: {research.researchTime}s
                      </div>
                    </div>

                    {research.prerequisites.length > 0 && (
                      <div className="text-xs text-yellow-400 mb-2">
                        Requires:{' '}
                        {research.prerequisites.map(prereq => {
                          const prereqResearch = RESEARCH_TREE.find(r => r.id === prereq);
                          const hasPrereq = researchState.completedResearch.has(prereq);
                          return (
                            <span
                              key={prereq}
                              className={hasPrereq ? 'text-green-400' : 'text-red-400'}
                            >
                              {prereqResearch?.name}
                              {research.prerequisites.indexOf(prereq) < research.prerequisites.length - 1 ? ', ' : ''}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    {status === 'available' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartResearch(research.id);
                        }}
                        disabled={!affordable || researchState.activeResearch !== null}
                        className={`w-full py-2 rounded font-semibold ${
                          affordable && !researchState.activeResearch
                            ? 'bg-green-600 hover:bg-green-700 text-white'
                            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                      >
                        {!affordable
                          ? 'Cannot Afford'
                          : researchState.activeResearch
                          ? 'Already Researching'
                          : 'Start Research'}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Research Details Panel */}
          <div className="w-1/3 bg-gray-800 border-l-2 border-gray-700 p-4 overflow-y-auto">
            {selectedResearch ? (
              <>
                <div className={`text-sm font-semibold mb-2 ${getCategoryColor(selectedResearch.category)}`}>
                  {selectedResearch.category.toUpperCase()}
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">{selectedResearch.name}</h3>

                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Description</h4>
                  <p className="text-gray-300">{selectedResearch.description}</p>
                </div>

                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Cost</h4>
                  <div className="space-y-1 text-white">
                    {selectedResearch.cost.gold > 0 && (
                      <div className="flex justify-between">
                        <span className="text-yellow-500">Gold:</span>
                        <span>{selectedResearch.cost.gold}</span>
                      </div>
                    )}
                    {selectedResearch.cost.wood > 0 && (
                      <div className="flex justify-between">
                        <span className="text-yellow-700">Wood:</span>
                        <span>{selectedResearch.cost.wood}</span>
                      </div>
                    )}
                    {selectedResearch.cost.stone > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Stone:</span>
                        <span>{selectedResearch.cost.stone}</span>
                      </div>
                    )}
                    {selectedResearch.cost.food > 0 && (
                      <div className="flex justify-between">
                        <span className="text-green-500">Food:</span>
                        <span>{selectedResearch.cost.food}</span>
                      </div>
                    )}
                  </div>
                  <div className="mt-2 text-sm text-gray-400">
                    Research Time: {selectedResearch.researchTime} seconds
                  </div>
                </div>

                {selectedResearch.prerequisites.length > 0 && (
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold text-gray-400 mb-2">Prerequisites</h4>
                    <ul className="space-y-1">
                      {selectedResearch.prerequisites.map(prereq => {
                        const prereqResearch = RESEARCH_TREE.find(r => r.id === prereq);
                        const hasPrereq = researchState.completedResearch.has(prereq);
                        return (
                          <li key={prereq} className="flex items-center gap-2">
                            <span className={hasPrereq ? 'text-green-400' : 'text-red-400'}>
                              {hasPrereq ? '✓' : '✗'}
                            </span>
                            <span className="text-gray-300">{prereqResearch?.name}</span>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}

                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Effects</h4>
                  <ul className="space-y-2">
                    {selectedResearch.effects.map((effect, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-green-400 mt-1">•</span>
                        <span className="text-gray-300">{effect.description}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="pt-4 border-t border-gray-700">
                  <h4 className="text-sm font-semibold text-gray-400 mb-2">Status</h4>
                  <div className="text-white font-semibold">
                    {getResearchStatus(selectedResearch).toUpperCase()}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center text-gray-400 mt-10">
                <p>Select a research to view details</p>
              </div>
            )}
          </div>
        </div>

        {/* Stats Footer */}
        <div className="bg-gray-800 border-t-2 border-gray-700 p-3 flex gap-6 text-sm">
          <div>
            <span className="text-gray-400">Completed Research:</span>
            <span className="text-white ml-2 font-semibold">
              {researchState.completedResearch.size} / {RESEARCH_TREE.length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Available Research:</span>
            <span className="text-green-400 ml-2 font-semibold">
              {researchStore.getAvailableResearch().length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResearchPanel;
