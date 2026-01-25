import React, { useEffect } from 'react';
import { useGameStore } from '../store/gameStore';
import Building from '../buildings/Building';
import Resource from './Resource';
import Unit from './Unit';

export const Game: React.FC = () => {
  const buildings = useGameStore((state) => state.buildings);
  const resources = useGameStore((state) => state.resources);
  const units = useGameStore((state) => state.units);
  const addBuilding = useGameStore((state) => state.addBuilding);
  const addResource = useGameStore((state) => state.addResource);

  useEffect(() => {
    addBuilding({
      id: 'building_1',
      type: 'town_hall',
      position: { x: 100, y: 100 },
      ownerId: 'player_1',
      productionQueue: [],
    });

    addResource({
      id: 'resource_1',
      type: 'gold',
      position: { x: 400, y: 200 },
      amount: 1000,
    });

    addResource({
      id: 'resource_2',
      type: 'wood',
      position: { x: 600, y: 300 },
      amount: 800,
    });

    addResource({
      id: 'resource_3',
      type: 'stone',
      position: { x: 300, y: 400 },
      amount: 500,
    });
  }, [addBuilding, addResource]);

  return (
    <div className="relative w-full h-screen bg-green-900 overflow-hidden">
      <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white p-4 rounded-lg z-10">
        <h1 className="text-2xl font-bold mb-2">Orca RTS</h1>
        <div className="text-sm">
          <div>Buildings: {buildings.size}</div>
          <div>Units: {units.size}</div>
          <div>Resources: {resources.size}</div>
        </div>
        <div className="mt-4 text-xs text-gray-300">
          <p>1. Click "Set Rally Point" on the building</p>
          <p>2. Click on a resource (gold mine, etc.)</p>
          <p>3. Click "Spawn Worker"</p>
          <p>4. Worker will auto-gather from the resource!</p>
        </div>
      </div>

      <div className="relative w-full h-full">
        {Array.from(resources.values()).map((resource) => (
          <Resource key={resource.id} resource={resource} />
        ))}

        {Array.from(buildings.values()).map((building) => (
          <Building key={building.id} building={building} />
        ))}

        {Array.from(units.values()).map((unit) => (
          <Unit key={unit.id} unit={unit} />
        ))}
      </div>

      <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white p-3 rounded-lg z-10 text-xs">
        <div className="font-bold mb-2">Legend:</div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-4 h-4 bg-yellow-400 rounded-full"></div>
          <span>Resource Rally Point</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-4 h-4 bg-white rounded-full"></div>
          <span>Regular Rally Point</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-600 rounded-full border-2 border-yellow-400"></div>
          <span>Gathering Worker</span>
        </div>
      </div>
    </div>
  );
};

export default Game;
