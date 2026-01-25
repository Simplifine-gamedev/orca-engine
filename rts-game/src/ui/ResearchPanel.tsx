'use client';

import React, { useState } from 'react';
import { useResearchStore } from '../store/researchStore';
import { blacksmithResearch, canResearch } from '../buildings/buildingModels';
import { ResearchTech, ResearchItem } from '../types/research';

export function ResearchPanel() {
  const {
    completedResearch,
    activeResearch,
    availableGold,
    availableFood,
    startResearch,
    cancelResearch,
  } = useResearchStore();
  
  const [hoveredTech, setHoveredTech] = useState<ResearchTech | null>(null);
  
  const handleResearchClick = (techId: ResearchTech) => {
    if (activeResearch) {
      // Can't start new research while one is active
      return;
    }
    startResearch(techId);
  };
  
  const researchItems = Object.values(blacksmithResearch);
  
  return (
    <div className="research-panel bg-gray-800 border border-gray-700 rounded-lg p-4 w-full max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-red-600 rounded-md flex items-center justify-center">
            <span className="text-white font-bold">🔨</span>
          </div>
          <h2 className="text-xl font-bold text-white">Blacksmith Research</h2>
        </div>
        
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-1">
            <span className="text-yellow-400">💰</span>
            <span className="text-white">{availableGold}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-green-400">🌾</span>
            <span className="text-white">{availableFood}</span>
          </div>
        </div>
      </div>
      
      {/* Active Research */}
      {activeResearch && (
        <div className="mb-4 p-3 bg-blue-900 bg-opacity-30 border border-blue-500 rounded-md">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              <span className="text-white font-semibold">
                Researching: {blacksmithResearch[activeResearch.techId].name}
              </span>
            </div>
            <button
              onClick={cancelResearch}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition"
            >
              Cancel
            </button>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-blue-500 to-blue-400 h-full transition-all duration-200"
              style={{ width: `${activeResearch.progress}%` }}
            >
              <span className="text-xs text-white font-semibold px-2 leading-4">
                {Math.floor(activeResearch.progress)}%
              </span>
            </div>
          </div>
        </div>
      )}
      
      {/* Research Items Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {researchItems.map((tech) => {
          const isCompleted = completedResearch.includes(tech.id);
          const isResearching = activeResearch?.techId === tech.id;
          const canStartResearch = !isCompleted && 
                                   !activeResearch && 
                                   canResearch(tech.id, completedResearch);
          const hasEnoughResources = availableGold >= tech.cost.gold && 
                                     availableFood >= (tech.cost.food || 0);
          const isLocked = !canStartResearch && !isCompleted;
          
          return (
            <div
              key={tech.id}
              className={`
                research-item relative p-3 rounded-lg border-2 transition-all cursor-pointer
                ${isCompleted ? 'bg-green-900 bg-opacity-30 border-green-500' : 
                  isResearching ? 'bg-blue-900 bg-opacity-30 border-blue-500' :
                  isLocked ? 'bg-gray-900 bg-opacity-50 border-gray-600 cursor-not-allowed opacity-50' :
                  hasEnoughResources ? 'bg-gray-800 border-gray-600 hover:border-yellow-500' :
                  'bg-gray-800 border-red-600 opacity-75'}
              `}
              onClick={() => canStartResearch && hasEnoughResources && handleResearchClick(tech.id)}
              onMouseEnter={() => setHoveredTech(tech.id)}
              onMouseLeave={() => setHoveredTech(null)}
            >
              {/* Icon placeholder */}
              <div className={`
                w-12 h-12 rounded-md mb-2 flex items-center justify-center text-2xl
                ${isCompleted ? 'bg-green-600' : 
                  isResearching ? 'bg-blue-600' : 
                  'bg-gray-700'}
              `}>
                {isCompleted ? '✓' : 
                 isResearching ? '⏳' : 
                 isLocked ? '🔒' : '⚔️'}
              </div>
              
              <h3 className="text-white font-semibold text-sm mb-1">
                {tech.name}
              </h3>
              
              <p className="text-gray-400 text-xs mb-2 line-clamp-2">
                {tech.description}
              </p>
              
              {/* Cost */}
              <div className="flex gap-2 text-xs">
                <span className={availableGold >= tech.cost.gold ? 'text-yellow-400' : 'text-red-400'}>
                  💰 {tech.cost.gold}
                </span>
                {tech.cost.food && (
                  <span className={availableFood >= tech.cost.food ? 'text-green-400' : 'text-red-400'}>
                    🌾 {tech.cost.food}
                  </span>
                )}
              </div>
              
              {/* Requirements */}
              {tech.requirements && tech.requirements.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <p className="text-xs text-gray-500">
                    Requires: {tech.requirements.map(req => 
                      blacksmithResearch[req].name
                    ).join(', ')}
                  </p>
                </div>
              )}
              
              {/* Hover tooltip */}
              {hoveredTech === tech.id && (
                <div className="absolute z-10 left-full ml-2 top-0 w-64 p-3 bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
                  <h4 className="text-white font-bold mb-2">{tech.name}</h4>
                  <p className="text-gray-300 text-sm mb-2">{tech.description}</p>
                  <div className="text-xs text-blue-400 mb-2">
                    ⏱️ Research Time: {tech.researchTime}s
                  </div>
                  <div className="text-xs text-green-400 font-semibold">
                    Effect: {tech.effects.description}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-700 flex gap-4 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-600 rounded" />
          <span>Completed</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-blue-600 rounded" />
          <span>Researching</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-gray-700 rounded" />
          <span>Available</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-gray-900 rounded opacity-50" />
          <span>Locked</span>
        </div>
      </div>
    </div>
  );
}

export function ResearchButton({ buildingId }: { buildingId: string }) {
  const [showPanel, setShowPanel] = useState(false);
  
  return (
    <div>
      <button
        onClick={() => setShowPanel(!showPanel)}
        className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white font-semibold rounded-md transition"
      >
        🔨 Research Technologies
      </button>
      
      {showPanel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="relative">
            <button
              onClick={() => setShowPanel(false)}
              className="absolute -top-2 -right-2 w-8 h-8 bg-red-600 hover:bg-red-700 text-white rounded-full z-10"
            >
              ✕
            </button>
            <ResearchPanel />
          </div>
        </div>
      )}
    </div>
  );
}
