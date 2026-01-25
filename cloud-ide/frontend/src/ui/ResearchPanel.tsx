'use client'

import React, { useEffect, useState } from 'react';
import { ResearchTech, GameResources } from '../types/game';
import { researchStore } from '../store/researchStore';
import { getBlacksmithTechs } from '../data/researchTechs';

interface ResearchPanelProps {
  buildingId: string;
  buildingName: string;
  playerResources: GameResources;
  onStartResearch?: (techId: string) => void;
  onCancelResearch?: () => void;
}

export const ResearchPanel: React.FC<ResearchPanelProps> = ({
  buildingId,
  buildingName,
  playerResources,
  onStartResearch,
  onCancelResearch,
}) => {
  const [researchState, setResearchState] = useState(researchStore.getState());
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const unsubscribe = researchStore.subscribe(() => {
      setResearchState(researchStore.getState());
      setProgress(researchStore.getResearchProgress());
    });

    return unsubscribe;
  }, []);

  const availableTechs = researchStore.getTechsForBuilding(buildingId);
  const activeResearch = researchState.activeResearch;

  const handleStartResearch = (techId: string) => {
    if (researchStore.canStartResearch(techId, playerResources)) {
      researchStore.startResearch(techId, playerResources);
      onStartResearch?.(techId);
    }
  };

  const handleCancelResearch = () => {
    researchStore.cancelResearch();
    onCancelResearch?.();
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 max-w-2xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">{buildingName} Research</h2>
        <div className="flex gap-4 text-sm">
          <span className="text-yellow-400">💰 {playerResources.gold}</span>
          <span className="text-amber-600">🪵 {playerResources.wood}</span>
          <span className="text-gray-400">🪨 {playerResources.stone}</span>
        </div>
      </div>

      {/* Active Research */}
      {activeResearch && (
        <div className="bg-gray-900 rounded-lg p-4 mb-4 border-2 border-blue-500">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-white font-semibold">
              Currently Researching: {researchState.availableTechs.find(t => t.id === activeResearch.techId)?.name}
            </h3>
            <button
              onClick={handleCancelResearch}
              className="text-red-400 hover:text-red-300 text-sm"
            >
              Cancel
            </button>
          </div>
          <div className="w-full h-4 bg-gray-700 rounded-full overflow-hidden mb-2">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-gray-400 text-sm">{Math.round(progress)}% Complete</div>
        </div>
      )}

      {/* Available Research */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {availableTechs.length === 0 ? (
          <div className="text-gray-400 text-center py-8">
            No research available for this building
          </div>
        ) : (
          availableTechs.map(tech => {
            const isCompleted = researchStore.isResearchCompleted(tech.id);
            const canResearch = researchStore.canStartResearch(tech.id, playerResources);
            const hasPrerequisites = tech.prerequisites.every(prereq =>
              researchStore.isResearchCompleted(prereq)
            );

            return (
              <ResearchTechCard
                key={tech.id}
                tech={tech}
                isCompleted={isCompleted}
                canResearch={canResearch}
                hasPrerequisites={hasPrerequisites}
                isResearching={activeResearch?.techId === tech.id}
                onStartResearch={() => handleStartResearch(tech.id)}
              />
            );
          })
        )}
      </div>
    </div>
  );
};

interface ResearchTechCardProps {
  tech: ResearchTech;
  isCompleted: boolean;
  canResearch: boolean;
  hasPrerequisites: boolean;
  isResearching: boolean;
  onStartResearch: () => void;
}

const ResearchTechCard: React.FC<ResearchTechCardProps> = ({
  tech,
  isCompleted,
  canResearch,
  hasPrerequisites,
  isResearching,
  onStartResearch,
}) => {
  return (
    <div
      className={`
        bg-gray-900 rounded-lg p-3 transition-all
        ${isCompleted ? 'opacity-60 border-2 border-green-600' : ''}
        ${canResearch && !isCompleted && !isResearching ? 'hover:bg-gray-850 cursor-pointer border-2 border-blue-500' : ''}
        ${!hasPrerequisites ? 'opacity-40' : ''}
      `}
      onClick={() => {
        if (canResearch && !isCompleted && !isResearching) {
          onStartResearch();
        }
      }}
    >
      <div className="flex items-start gap-3">
        {/* Tech Icon */}
        <div className="w-12 h-12 bg-gray-800 rounded flex items-center justify-center flex-shrink-0">
          {tech.icon ? (
            <img src={tech.icon} alt={tech.name} className="w-8 h-8" />
          ) : (
            <span className="text-2xl">⚔️</span>
          )}
        </div>

        {/* Tech Info */}
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-white font-semibold">{tech.name}</h3>
            {isCompleted && (
              <span className="text-green-400 text-sm">✓ Completed</span>
            )}
          </div>
          
          <p className="text-gray-400 text-sm mb-2">{tech.description}</p>

          {/* Effects */}
          <div className="space-y-1 mb-2">
            {tech.effects.map((effect, idx) => (
              <div key={idx} className="text-xs text-blue-300">
                • {effect.description}
              </div>
            ))}
          </div>

          {/* Prerequisites */}
          {tech.prerequisites.length > 0 && (
            <div className="text-xs text-gray-500 mb-2">
              Requires: {tech.prerequisites.map(p => p.replace('_', ' ')).join(', ')}
            </div>
          )}

          {/* Cost and Time */}
          <div className="flex items-center justify-between">
            <div className="flex gap-3 text-sm">
              {tech.cost.gold && (
                <span className={canResearch ? 'text-yellow-400' : 'text-red-400'}>
                  💰 {tech.cost.gold}
                </span>
              )}
              {tech.cost.wood && (
                <span className={canResearch ? 'text-amber-600' : 'text-red-400'}>
                  🪵 {tech.cost.wood}
                </span>
              )}
              {tech.cost.stone && (
                <span className={canResearch ? 'text-gray-400' : 'text-red-400'}>
                  🪨 {tech.cost.stone}
                </span>
              )}
            </div>
            <span className="text-gray-500 text-xs">⏱️ {tech.researchTime}s</span>
          </div>

          {/* Action Status */}
          {!isCompleted && !hasPrerequisites && (
            <div className="text-red-400 text-xs mt-2">
              Missing prerequisites
            </div>
          )}
          {!isCompleted && hasPrerequisites && !canResearch && (
            <div className="text-red-400 text-xs mt-2">
              Insufficient resources
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
