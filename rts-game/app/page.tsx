'use client';

import React, { Suspense, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import { Building } from '../src/buildings/Building';
import { ResearchPanel, ResearchButton } from '../src/ui/ResearchPanel';
import { useBuildingStore } from '../src/store/buildingStore';
import { useResearchStore } from '../src/store/researchStore';
import { buildingModels } from '../src/buildings/buildingModels';

export default function OrcaRTS() {
  const { buildings, addBuilding, selectedBuilding } = useBuildingStore();
  const { addResources, availableGold, availableFood } = useResearchStore();
  const [showResearchPanel, setShowResearchPanel] = useState(false);
  const [buildMode, setBuildMode] = useState<import('../src/types/building').BuildingType | null>(null);
  
  // Initialize with a blacksmith if none exists
  React.useEffect(() => {
    if (buildings.length === 0) {
      addBuilding('blacksmith', { x: 0, y: 0 });
    }
  }, []);
  
  const handleCanvasClick = (event: any) => {
    if (buildMode && event.point) {
      addBuilding(buildMode, { x: event.point.x, y: event.point.z });
      setBuildMode(null);
    }
  };
  
  return (
    <div className="w-full h-screen flex flex-col bg-gray-900">
      {/* Top Bar */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-white">Orca RTS</h1>
            <div className="h-6 w-px bg-gray-600" />
            <div className="flex gap-4 text-sm">
              <div className="flex items-center gap-1">
                <span className="text-yellow-400">💰 Gold:</span>
                <span className="text-white font-semibold">{availableGold}</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-green-400">🌾 Food:</span>
                <span className="text-white font-semibold">{availableFood}</span>
              </div>
            </div>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={() => addResources(100, 50)}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded transition"
            >
              + Add Resources
            </button>
            <button
              onClick={() => setShowResearchPanel(!showResearchPanel)}
              className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded transition"
            >
              🔨 Research
            </button>
          </div>
        </div>
      </div>
      
      {/* Main Game View */}
      <div className="flex-1 flex">
        {/* 3D Viewport */}
        <div className="flex-1 relative">
          <Canvas shadows onClick={handleCanvasClick}>
            <PerspectiveCamera makeDefault position={[15, 15, 15]} />
            <OrbitControls 
              enablePan={true}
              enableZoom={true}
              maxPolarAngle={Math.PI / 2.5}
            />
            
            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight
              position={[10, 20, 10]}
              intensity={1}
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
            />
            <pointLight position={[0, 10, 0]} intensity={0.3} />
            
            {/* Ground Grid */}
            <Grid args={[50, 50]} cellColor="#444" sectionColor="#666" />
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
              <planeGeometry args={[100, 100]} />
              <meshStandardMaterial color="#1a1a2e" />
            </mesh>
            
            {/* Buildings */}
            <Suspense fallback={null}>
              {buildings.map((building) => (
                <Building key={building.instanceId} building={building} />
              ))}
            </Suspense>
          </Canvas>
          
          {buildMode && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg">
              Click on the map to place {buildingModels[buildMode]?.name}
              <button 
                onClick={() => setBuildMode(null)}
                className="ml-4 px-2 py-1 bg-red-500 hover:bg-red-600 rounded text-sm"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
        
        {/* Side Panel */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 p-4 overflow-y-auto">
          <h2 className="text-xl font-bold text-white mb-4">Building Menu</h2>
          
          {/* Building buttons */}
          <div className="space-y-2 mb-6">
            {Object.values(buildingModels).map((building) => (
              <button
                key={building.id}
                onClick={() => setBuildMode(building.id)}
                className={`w-full p-3 rounded-lg text-left transition ${
                  buildMode === building.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 hover:bg-gray-600 text-white'
                }`}
              >
                <div className="font-semibold">{building.name}</div>
                <div className="text-xs text-gray-300 mt-1">
                  💰 {building.cost.gold} | 🪵 {building.cost.wood} | 🪨 {building.cost.stone}
                </div>
              </button>
            ))}
          </div>
          
          {/* Selected Building Info */}
          {selectedBuilding && (
            <div className="border-t border-gray-700 pt-4">
              <h3 className="text-lg font-bold text-white mb-2">
                {selectedBuilding.name}
              </h3>
              <div className="space-y-2 text-sm text-gray-300">
                <div>Health: {selectedBuilding.health}/{selectedBuilding.hitPoints}</div>
                <div>Position: ({selectedBuilding.position.x.toFixed(1)}, {selectedBuilding.position.y.toFixed(1)})</div>
                {selectedBuilding.constructionProgress < 100 && (
                  <div>Construction: {selectedBuilding.constructionProgress.toFixed(0)}%</div>
                )}
                
                {selectedBuilding.id === 'blacksmith' && selectedBuilding.constructionProgress >= 100 && (
                  <button
                    onClick={() => setShowResearchPanel(true)}
                    className="w-full mt-4 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded transition"
                  >
                    🔨 Open Research Panel
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Research Panel Modal */}
      {showResearchPanel && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="relative">
            <button
              onClick={() => setShowResearchPanel(false)}
              className="absolute -top-3 -right-3 w-10 h-10 bg-red-600 hover:bg-red-700 text-white rounded-full text-xl font-bold shadow-lg z-10"
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
