import React, { useState, useEffect } from 'react';
import { WorldMapSelector } from './WorldMapSelector';
import { MapPreset } from '../types/maps';

interface Player {
  id: string;
  name: string;
  team: number;
  ready: boolean;
  isHost: boolean;
}

interface GameLobbyProps {
  lobbyId: string;
  localPlayerId: string;
  onStartGame: (mapId: string) => void;
  onLeaveLobby: () => void;
}

export const GameLobby: React.FC<GameLobbyProps> = ({
  lobbyId,
  localPlayerId,
  onStartGame,
  onLeaveLobby,
}) => {
  const [players, setPlayers] = useState<Player[]>([
    {
      id: localPlayerId,
      name: 'You',
      team: 1,
      ready: false,
      isHost: true,
    },
  ]);
  const [selectedMap, setSelectedMap] = useState<MapPreset | null>(null);
  const [showMapSelector, setShowMapSelector] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{ player: string; message: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [gameSettings, setGameSettings] = useState({
    maxPlayers: 4,
    startingResources: 'normal',
    gameSpeed: 'normal',
    fogOfWar: true,
  });

  const localPlayer = players.find((p) => p.id === localPlayerId);
  const isHost = localPlayer?.isHost || false;
  const allPlayersReady = players.every((p) => p.ready || p.isHost);
  const canStartGame = isHost && allPlayersReady && selectedMap && players.length >= 2;

  const handleMapSelect = (map: MapPreset) => {
    setSelectedMap(map);
    setShowMapSelector(false);
  };

  const handleToggleReady = () => {
    setPlayers((prev) =>
      prev.map((p) =>
        p.id === localPlayerId ? { ...p, ready: !p.ready } : p
      )
    );
  };

  const handleTeamChange = (playerId: string, newTeam: number) => {
    if (!isHost) return;
    setPlayers((prev) =>
      prev.map((p) => (p.id === playerId ? { ...p, team: newTeam } : p))
    );
  };

  const handleSendChat = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    setChatMessages((prev) => [
      ...prev,
      { player: localPlayer?.name || 'Unknown', message: chatInput },
    ]);
    setChatInput('');
  };

  const handleStartGame = () => {
    if (canStartGame && selectedMap) {
      onStartGame(selectedMap.id);
    }
  };

  return (
    <div className="game-lobby min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6 flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Game Lobby</h1>
            <p className="text-gray-400">Lobby ID: {lobbyId}</p>
          </div>
          <button
            onClick={onLeaveLobby}
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Leave Lobby
          </button>
        </div>

        {showMapSelector ? (
          <div className="mb-6">
            <WorldMapSelector
              selectedMapId={selectedMap?.id || null}
              onMapSelect={handleMapSelect}
              maxPlayers={gameSettings.maxPlayers}
            />
            <button
              onClick={() => setShowMapSelector(false)}
              className="mt-4 bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-lg transition-colors"
            >
              Close Map Selection
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-white">Selected Map</h2>
                  {isHost && (
                    <button
                      onClick={() => setShowMapSelector(true)}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      Change Map
                    </button>
                  )}
                </div>

                {selectedMap ? (
                  <div className="bg-gray-750 rounded-lg p-4 border border-gray-600">
                    <div className="flex gap-4">
                      <div className="w-48 h-48 bg-gray-900 rounded flex items-center justify-center text-6xl">
                        {selectedMap.terrain === 'mixed' && '🏝️'}
                        {selectedMap.terrain === 'desert' && '🏜️'}
                        {selectedMap.terrain === 'snow' && '❄️'}
                        {selectedMap.terrain === 'volcanic' && '🌋'}
                        {selectedMap.terrain === 'grass' && '🌿'}
                        {selectedMap.terrain === 'urban' && '🏙️'}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-2xl font-bold text-white mb-2">{selectedMap.name}</h3>
                        <p className="text-gray-400 mb-3">{selectedMap.description}</p>
                        <div className="flex flex-wrap gap-2">
                          <span className="bg-gray-800 px-3 py-1 rounded text-sm text-gray-300">
                            Size: {selectedMap.size.width}x{selectedMap.size.height}
                          </span>
                          <span className="bg-gray-800 px-3 py-1 rounded text-sm text-gray-300">
                            Max Players: {selectedMap.maxPlayers}
                          </span>
                          <span className="bg-gray-800 px-3 py-1 rounded text-sm text-gray-300">
                            Difficulty: {selectedMap.difficulty}
                          </span>
                          <span className="bg-gray-800 px-3 py-1 rounded text-sm text-gray-300">
                            Layout: {selectedMap.layout}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-750 rounded-lg p-12 text-center border border-gray-600">
                    <p className="text-gray-400 text-lg mb-4">No map selected</p>
                    {isHost && (
                      <button
                        onClick={() => setShowMapSelector(true)}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors"
                      >
                        Select a Map
                      </button>
                    )}
                    {!isHost && (
                      <p className="text-gray-500">Waiting for host to select map...</p>
                    )}
                  </div>
                )}
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 className="text-2xl font-bold text-white mb-4">Players ({players.length})</h2>
                <div className="space-y-2">
                  {players.map((player) => (
                    <div
                      key={player.id}
                      className="flex items-center justify-between bg-gray-750 p-4 rounded-lg border border-gray-600"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center text-xl">
                          👤
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-white font-semibold">{player.name}</span>
                            {player.isHost && (
                              <span className="bg-yellow-500 text-black text-xs px-2 py-1 rounded">
                                HOST
                              </span>
                            )}
                            {player.ready && !player.isHost && (
                              <span className="bg-green-500 text-white text-xs px-2 py-1 rounded">
                                READY
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <select
                          value={player.team}
                          onChange={(e) => handleTeamChange(player.id, parseInt(e.target.value))}
                          disabled={!isHost}
                          className="bg-gray-700 text-white px-3 py-2 rounded border border-gray-600 disabled:opacity-50"
                        >
                          <option value={1}>Team 1</option>
                          <option value={2}>Team 2</option>
                          <option value={3}>Team 3</option>
                          <option value={4}>Team 4</option>
                        </select>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {isHost && (
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h2 className="text-2xl font-bold text-white mb-4">Game Settings</h2>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-gray-400 mb-2">Starting Resources</label>
                      <select
                        value={gameSettings.startingResources}
                        onChange={(e) =>
                          setGameSettings({ ...gameSettings, startingResources: e.target.value })
                        }
                        className="w-full bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
                      >
                        <option value="low">Low</option>
                        <option value="normal">Normal</option>
                        <option value="high">High</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-gray-400 mb-2">Game Speed</label>
                      <select
                        value={gameSettings.gameSpeed}
                        onChange={(e) =>
                          setGameSettings({ ...gameSettings, gameSpeed: e.target.value })
                        }
                        className="w-full bg-gray-700 text-white px-3 py-2 rounded border border-gray-600"
                      >
                        <option value="slow">Slow</option>
                        <option value="normal">Normal</option>
                        <option value="fast">Fast</option>
                      </select>
                    </div>
                    <div className="col-span-2">
                      <label className="flex items-center gap-2 text-white cursor-pointer">
                        <input
                          type="checkbox"
                          checked={gameSettings.fogOfWar}
                          onChange={(e) =>
                            setGameSettings({ ...gameSettings, fogOfWar: e.target.checked })
                          }
                          className="w-5 h-5"
                        />
                        Fog of War
                      </label>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-6">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 className="text-2xl font-bold text-white mb-4">Chat</h2>
                <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto mb-4">
                  {chatMessages.length === 0 ? (
                    <p className="text-gray-500 text-center">No messages yet</p>
                  ) : (
                    <div className="space-y-2">
                      {chatMessages.map((msg, idx) => (
                        <div key={idx} className="text-sm">
                          <span className="text-blue-400 font-semibold">{msg.player}:</span>
                          <span className="text-gray-300 ml-2">{msg.message}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <form onSubmit={handleSendChat} className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Type a message..."
                    className="flex-1 bg-gray-700 text-white px-3 py-2 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
                  />
                  <button
                    type="submit"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors"
                  >
                    Send
                  </button>
                </form>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                {isHost ? (
                  <button
                    onClick={handleStartGame}
                    disabled={!canStartGame}
                    className={`w-full py-4 rounded-lg text-xl font-bold transition-colors ${
                      canStartGame
                        ? 'bg-green-600 hover:bg-green-700 text-white'
                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    }`}
                  >
                    {!selectedMap
                      ? 'Select a Map'
                      : !allPlayersReady
                      ? 'Waiting for Players...'
                      : players.length < 2
                      ? 'Need More Players'
                      : 'Start Game'}
                  </button>
                ) : (
                  <button
                    onClick={handleToggleReady}
                    className={`w-full py-4 rounded-lg text-xl font-bold transition-colors ${
                      localPlayer?.ready
                        ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {localPlayer?.ready ? 'Not Ready' : 'Ready'}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
