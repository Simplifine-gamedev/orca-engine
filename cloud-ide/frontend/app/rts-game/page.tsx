'use client';

import React from 'react';
import { GameProvider, useGame } from './src/game/GameContext';
import { ResourceBar } from './src/ui/ResourceBar';
import { WorkerBuildPanel } from './src/ui/WorkerBuildPanel';
import { BuildingList } from './src/buildings/Building';

function GameContent() {
  const { gameState } = useGame();

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <div className="bg-gray-950 border-b-2 border-gray-800 p-4">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2">⚔️ Orca RTS - Resource Management Demo</h1>
          <p className="text-gray-400">
            A demo showcasing improved resource system clarity for players
          </p>
        </div>
      </div>

      {/* Resource Bar */}
      <ResourceBar />

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-6">
        {/* Info Panel */}
        <div className="mb-6 p-4 bg-blue-900/30 rounded-lg border-2 border-blue-700">
          <h2 className="text-lg font-bold text-blue-400 mb-2">🎮 How to Play</h2>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• <strong>Resources</strong> generate automatically every second (shown as +X/sec)</li>
            <li>• <strong>Hover</strong> over resources to see what they're used for</li>
            <li>• <strong>Green border</strong> on buildings means you can afford them</li>
            <li>• <strong>Red border</strong> means you need more resources</li>
            <li>• <strong>Click</strong> buildings to see detailed information and production rates</li>
            <li>• Build resource buildings to increase your income!</li>
          </ul>
        </div>

        {/* Your Buildings */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <span>🏛️</span> Your Buildings ({gameState.buildings.length})
          </h2>
          <BuildingList buildings={gameState.buildings} />
        </div>

        {/* Build Panel */}
        <WorkerBuildPanel />
      </div>

      {/* Footer with Issue Resolution Info */}
      <div className="mt-12 border-t-2 border-gray-800 bg-gray-950 p-6">
        <div className="max-w-7xl mx-auto">
          <h3 className="text-xl font-bold text-white mb-3">✅ Issue Resolution: ORC-160</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="p-3 bg-green-900/30 rounded border border-green-700">
              <h4 className="font-bold text-green-400 mb-2">1. Tutorial/Tooltips ✓</h4>
              <p className="text-gray-300">Hover over any resource or building to see detailed tooltips explaining what they do and what they're used for.</p>
            </div>
            <div className="p-3 bg-green-900/30 rounded border border-green-700">
              <h4 className="font-bold text-green-400 mb-2">2. Resource Costs ✓</h4>
              <p className="text-gray-300">All building cards clearly show resource costs with color-coding (green = can afford, red = cannot afford).</p>
            </div>
            <div className="p-3 bg-green-900/30 rounded border border-green-700">
              <h4 className="font-bold text-green-400 mb-2">3. Affordability Highlighting ✓</h4>
              <p className="text-gray-300">Buildings have clear borders and badges showing "READY TO BUILD" or "CAN'T AFFORD" status.</p>
            </div>
            <div className="p-3 bg-green-900/30 rounded border border-green-700">
              <h4 className="font-bold text-green-400 mb-2">4. Resource Income Indicators ✓</h4>
              <p className="text-gray-300">Every resource shows +X/sec income rate, and buildings display their production contributions.</p>
            </div>
            <div className="p-3 bg-green-900/30 rounded border border-green-700">
              <h4 className="font-bold text-green-400 mb-2">5. Resource Usage Info ✓</h4>
              <p className="text-gray-300">Tooltips explain what each resource is used for and how to generate more.</p>
            </div>
            <div className="p-3 bg-green-900/30 rounded border border-green-700">
              <h4 className="font-bold text-green-400 mb-2">Bonus: Visual Feedback ✓</h4>
              <p className="text-gray-300">Color-coded costs, hover effects, and clear production displays help players understand the system at a glance.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RTSGame() {
  return (
    <GameProvider>
      <GameContent />
    </GameProvider>
  );
}
