import React, { useState } from 'react';
import { MapPreset, MAP_PRESETS } from '../types/maps';

interface WorldMapSelectorProps {
  selectedMapId: string | null;
  onMapSelect: (map: MapPreset) => void;
  maxPlayers?: number;
}

export const WorldMapSelector: React.FC<WorldMapSelectorProps> = ({
  selectedMapId,
  onMapSelect,
  maxPlayers,
}) => {
  const [filter, setFilter] = useState<'all' | 'Easy' | 'Medium' | 'Hard'>('all');
  const [sortBy, setSortBy] = useState<'name' | 'size' | 'players'>('name');

  const filteredMaps = MAP_PRESETS.filter((map) => {
    if (filter !== 'all' && map.difficulty !== filter) return false;
    if (maxPlayers && map.maxPlayers < maxPlayers) return false;
    return true;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.name.localeCompare(b.name);
      case 'size':
        return (a.size.width * a.size.height) - (b.size.width * b.size.height);
      case 'players':
        return a.maxPlayers - b.maxPlayers;
      default:
        return 0;
    }
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy':
        return 'bg-green-500';
      case 'Medium':
        return 'bg-yellow-500';
      case 'Hard':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="world-map-selector bg-gray-800 rounded-lg p-6 w-full max-w-6xl">
      <div className="mb-6">
        <h2 className="text-3xl font-bold text-white mb-2">Select Map</h2>
        <p className="text-gray-400">Choose your battlefield</p>
      </div>

      <div className="flex gap-4 mb-6">
        <div className="flex gap-2">
          <label className="text-gray-300 font-medium">Difficulty:</label>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
          >
            <option value="all">All</option>
            <option value="Easy">Easy</option>
            <option value="Medium">Medium</option>
            <option value="Hard">Hard</option>
          </select>
        </div>

        <div className="flex gap-2">
          <label className="text-gray-300 font-medium">Sort by:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
          >
            <option value="name">Name</option>
            <option value="size">Size</option>
            <option value="players">Max Players</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredMaps.map((map) => (
          <div
            key={map.id}
            onClick={() => onMapSelect(map)}
            className={`
              map-card cursor-pointer rounded-lg overflow-hidden border-2 transition-all
              ${selectedMapId === map.id
                ? 'border-blue-500 shadow-lg shadow-blue-500/50 scale-105'
                : 'border-gray-700 hover:border-gray-500'
              }
            `}
          >
            <div className="relative h-48 bg-gray-900 overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-6xl">
                {map.terrain === 'mixed' && '🏝️'}
                {map.terrain === 'desert' && '🏜️'}
                {map.terrain === 'snow' && '❄️'}
                {map.terrain === 'volcanic' && '🌋'}
                {map.terrain === 'grass' && '🌿'}
                {map.terrain === 'urban' && '🏙️'}
              </div>
              {selectedMapId === map.id && (
                <div className="absolute top-2 right-2 bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                  ✓ Selected
                </div>
              )}
            </div>

            <div className="p-4 bg-gray-750">
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-xl font-bold text-white">{map.name}</h3>
                <span className={`${getDifficultyColor(map.difficulty)} text-white text-xs px-2 py-1 rounded`}>
                  {map.difficulty}
                </span>
              </div>

              <p className="text-gray-400 text-sm mb-3 line-clamp-2">
                {map.description}
              </p>

              <div className="flex flex-wrap gap-2 text-xs">
                <div className="bg-gray-800 px-2 py-1 rounded text-gray-300">
                  📏 {map.size.width}x{map.size.height}
                </div>
                <div className="bg-gray-800 px-2 py-1 rounded text-gray-300">
                  👥 {map.maxPlayers} players
                </div>
                <div className="bg-gray-800 px-2 py-1 rounded text-gray-300">
                  🗺️ {map.layout}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredMaps.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-400 text-lg">No maps match your filters</p>
        </div>
      )}
    </div>
  );
};
