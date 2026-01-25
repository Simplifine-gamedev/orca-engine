import React, { useState, useEffect } from 'react';
import {
  Research,
  ResearchCategory,
  ResearchStatus,
  ResearchProgress,
} from '../types/research';
import { ResearchStore } from '../store/researchStore';
import { GameStore } from '../store/gameStore';

interface ResearchPanelProps {
  researchStore: ResearchStore;
  gameStore: GameStore;
  onClose?: () => void;
}

export const ResearchPanel: React.FC<ResearchPanelProps> = ({
  researchStore,
  gameStore,
  onClose,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<ResearchCategory>(
    ResearchCategory.MILITARY
  );
  const [researches, setResearches] = useState<Research[]>([]);
  const [currentResearch, setCurrentResearch] = useState<ResearchProgress | null>(
    null
  );
  const [resources, setResources] = useState(gameStore.getResources());

  useEffect(() => {
    updateResearches();
    const interval = setInterval(() => {
      updateResearches();
      setResources(gameStore.getResources());
    }, 100);

    return () => clearInterval(interval);
  }, [selectedCategory]);

  const updateResearches = () => {
    const categoryResearches = researchStore.getResearchesByCategory(
      selectedCategory
    );
    setResearches(categoryResearches);
    setCurrentResearch(researchStore.getCurrentResearch());
  };

  const handleStartResearch = (researchId: string) => {
    const research = researchStore.getResearch(researchId);
    if (!research) return;

    const canAfford = gameStore.canAffordResources(research.cost);
    if (!canAfford) {
      alert('Not enough resources!');
      return;
    }

    const started = researchStore.startResearch(researchId, resources);
    if (started) {
      gameStore.deductResources(research.cost);
      updateResearches();
    }
  };

  const handleCancelResearch = () => {
    if (researchStore.cancelResearch()) {
      updateResearches();
    }
  };

  const getResearchStatusColor = (status: ResearchStatus): string => {
    switch (status) {
      case ResearchStatus.COMPLETED:
        return 'bg-green-500';
      case ResearchStatus.RESEARCHING:
        return 'bg-blue-500';
      case ResearchStatus.AVAILABLE:
        return 'bg-yellow-500';
      case ResearchStatus.LOCKED:
        return 'bg-gray-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getResearchStatusText = (status: ResearchStatus): string => {
    switch (status) {
      case ResearchStatus.COMPLETED:
        return 'Completed';
      case ResearchStatus.RESEARCHING:
        return 'Researching...';
      case ResearchStatus.AVAILABLE:
        return 'Available';
      case ResearchStatus.LOCKED:
        return 'Locked';
      default:
        return 'Unknown';
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const canAffordResearch = (research: Research): boolean => {
    return gameStore.canAffordResources(research.cost);
  };

  return (
    <div className="research-panel bg-gray-900 text-white p-6 rounded-lg shadow-2xl max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Research Tree</h1>
        {onClose && (
          <button
            onClick={onClose}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          >
            Close
          </button>
        )}
      </div>

      {/* Resources Display */}
      <div className="bg-gray-800 p-4 rounded mb-6">
        <div className="flex justify-around text-sm">
          <div className="flex items-center">
            <span className="text-yellow-400 mr-2">💰</span>
            <span>Gold: {Math.floor(resources.gold)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-brown-400 mr-2">🪵</span>
            <span>Wood: {Math.floor(resources.wood)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-gray-400 mr-2">🪨</span>
            <span>Stone: {Math.floor(resources.stone)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-green-400 mr-2">🌾</span>
            <span>Food: {Math.floor(resources.food)}</span>
          </div>
          <div className="flex items-center">
            <span className="text-purple-400 mr-2">✨</span>
            <span>Mana: {Math.floor(resources.mana)}</span>
          </div>
        </div>
      </div>

      {/* Current Research Progress */}
      {currentResearch && (
        <div className="bg-blue-900 p-4 rounded mb-6">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-lg font-semibold">
              Currently Researching:{' '}
              {researchStore.getResearch(currentResearch.researchId)?.name}
            </h3>
            <button
              onClick={handleCancelResearch}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
            >
              Cancel
            </button>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-4">
            <div
              className="bg-blue-500 h-4 rounded-full transition-all"
              style={{ width: `${currentResearch.progress * 100}%` }}
            />
          </div>
          <div className="text-sm mt-2 text-center">
            {Math.floor(currentResearch.progress * 100)}% -{' '}
            {formatTime(
              (currentResearch.totalTime / 1000) * (1 - currentResearch.progress)
            )}{' '}
            remaining
          </div>
        </div>
      )}

      {/* Category Tabs */}
      <div className="flex space-x-2 mb-6">
        {Object.values(ResearchCategory).map((category) => (
          <button
            key={category}
            onClick={() => setSelectedCategory(category)}
            className={`px-4 py-2 rounded capitalize ${
              selectedCategory === category
                ? 'bg-blue-600'
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
          >
            {category}
          </button>
        ))}
      </div>

      {/* Research List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
        {researches.map((research) => {
          const status = researchStore.getResearchStatus(research.id);
          const isAvailable = status === ResearchStatus.AVAILABLE;
          const canAfford = canAffordResearch(research);
          const isResearching =
            currentResearch?.researchId === research.id;

          return (
            <div
              key={research.id}
              className={`bg-gray-800 p-4 rounded border-2 ${
                isAvailable && canAfford
                  ? 'border-green-500'
                  : isAvailable
                  ? 'border-yellow-500'
                  : status === ResearchStatus.COMPLETED
                  ? 'border-blue-500'
                  : 'border-gray-600'
              }`}
            >
              {/* Research Header */}
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-lg font-semibold">{research.name}</h3>
                <span
                  className={`px-2 py-1 rounded text-xs ${getResearchStatusColor(
                    status
                  )}`}
                >
                  {getResearchStatusText(status)}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-300 mb-3">
                {research.description}
              </p>

              {/* Effects */}
              <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Effects:
                </h4>
                {research.effects.map((effect, idx) => (
                  <div key={idx} className="text-xs text-green-400">
                    • {effect.description}
                  </div>
                ))}
              </div>

              {/* Cost */}
              <div className="mb-3">
                <h4 className="text-xs font-semibold text-gray-400 mb-1">
                  Cost:
                </h4>
                <div className="flex flex-wrap gap-2 text-xs">
                  <span
                    className={
                      resources.gold >= research.cost.gold
                        ? 'text-yellow-400'
                        : 'text-red-400'
                    }
                  >
                    💰 {research.cost.gold}
                  </span>
                  {research.cost.wood && (
                    <span
                      className={
                        resources.wood >= research.cost.wood
                          ? 'text-brown-400'
                          : 'text-red-400'
                      }
                    >
                      🪵 {research.cost.wood}
                    </span>
                  )}
                  {research.cost.stone && (
                    <span
                      className={
                        resources.stone >= research.cost.stone
                          ? 'text-gray-400'
                          : 'text-red-400'
                      }
                    >
                      🪨 {research.cost.stone}
                    </span>
                  )}
                  {research.cost.food && (
                    <span
                      className={
                        resources.food >= research.cost.food
                          ? 'text-green-400'
                          : 'text-red-400'
                      }
                    >
                      🌾 {research.cost.food}
                    </span>
                  )}
                  {research.cost.mana && (
                    <span
                      className={
                        resources.mana >= research.cost.mana
                          ? 'text-purple-400'
                          : 'text-red-400'
                      }
                    >
                      ✨ {research.cost.mana}
                    </span>
                  )}
                </div>
              </div>

              {/* Time */}
              <div className="text-xs text-gray-400 mb-3">
                ⏱️ Research Time: {formatTime(research.researchTime)}
              </div>

              {/* Prerequisites */}
              {research.prerequisites.length > 0 && (
                <div className="mb-3">
                  <h4 className="text-xs font-semibold text-gray-400 mb-1">
                    Prerequisites:
                  </h4>
                  <div className="text-xs">
                    {research.prerequisites.map((prereqId) => {
                      const prereq = researchStore.getResearch(prereqId);
                      const isCompleted = researchStore
                        .getCompletedResearches()
                        .has(prereqId);
                      return (
                        <div
                          key={prereqId}
                          className={
                            isCompleted ? 'text-green-400' : 'text-red-400'
                          }
                        >
                          • {prereq?.name || prereqId}
                          {isCompleted ? ' ✓' : ' ✗'}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Action Button */}
              {isAvailable && !currentResearch && (
                <button
                  onClick={() => handleStartResearch(research.id)}
                  disabled={!canAfford}
                  className={`w-full py-2 rounded font-semibold ${
                    canAfford
                      ? 'bg-green-600 hover:bg-green-700'
                      : 'bg-gray-600 cursor-not-allowed'
                  }`}
                >
                  {canAfford ? 'Research' : 'Not Enough Resources'}
                </button>
              )}
              {isResearching && (
                <div className="w-full py-2 bg-blue-600 rounded text-center font-semibold">
                  Researching...
                </div>
              )}
              {status === ResearchStatus.COMPLETED && (
                <div className="w-full py-2 bg-green-600 rounded text-center font-semibold">
                  Completed ✓
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="mt-6 bg-gray-800 p-4 rounded">
        <h3 className="text-lg font-semibold mb-2">Research Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Completed:</span>{' '}
            <span className="text-green-400">
              {researchStore.getCompletedResearches().size}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Available:</span>{' '}
            <span className="text-yellow-400">
              {researchStore.getAvailableResearches().length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Total:</span>{' '}
            <span className="text-blue-400">
              {researchStore.getAllResearches().length}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Progress:</span>{' '}
            <span className="text-purple-400">
              {Math.floor(
                (researchStore.getCompletedResearches().size /
                  researchStore.getAllResearches().length) *
                  100
              )}
              %
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResearchPanel;
