/**
 * Example usage of the Game Lobby and Map Selection system
 * 
 * This file demonstrates how to integrate the lobby system into your game.
 */

import React, { useState } from 'react';
import { GameLobby } from './ui/GameLobby';
import { useGameLobby } from './hooks/useGameLobby';

const SERVER_URL = process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:3001';

export default function GameLobbyExample() {
  const [lobbyId, setLobbyId] = useState<string | null>(null);
  const [playerName, setPlayerName] = useState('');
  const [inLobby, setInLobby] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [gameConfig, setGameConfig] = useState<any>(null);

  // Create a new lobby
  const handleCreateLobby = async () => {
    if (!playerName.trim()) {
      alert('Please enter your name');
      return;
    }

    try {
      const response = await fetch(`${SERVER_URL}/api/lobbies/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hostName: playerName }),
      });

      const data = await response.json();
      setLobbyId(data.lobbyId);
      setInLobby(true);
    } catch (error) {
      console.error('Failed to create lobby:', error);
      alert('Failed to create lobby');
    }
  };

  // Join an existing lobby
  const handleJoinLobby = (id: string) => {
    if (!playerName.trim()) {
      alert('Please enter your name');
      return;
    }

    setLobbyId(id);
    setInLobby(true);
  };

  // Handle game start
  const handleGameStart = (mapId: string, settings: any) => {
    console.log('Game starting with map:', mapId, 'settings:', settings);
    setGameConfig({ mapId, settings });
    setGameStarted(true);
  };

  // Handle leaving lobby
  const handleLeaveLobby = () => {
    setInLobby(false);
    setLobbyId(null);
    setGameStarted(false);
  };

  // Render game started screen
  if (gameStarted && gameConfig) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Game Starting!</h1>
          <p className="text-xl mb-2">Map: {gameConfig.mapId}</p>
          <p className="text-gray-400">
            Loading game assets...
          </p>
          <div className="mt-8">
            <button
              onClick={handleLeaveLobby}
              className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Render lobby
  if (inLobby && lobbyId) {
    return (
      <LobbyWrapper
        lobbyId={lobbyId}
        playerName={playerName}
        onGameStart={handleGameStart}
        onLeaveLobby={handleLeaveLobby}
      />
    );
  }

  // Render main menu
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white flex items-center justify-center p-6">
      <div className="max-w-md w-full">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-2">Orca RTS</h1>
          <p className="text-gray-400">Multiplayer Strategy Game</p>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="mb-6">
            <label className="block text-gray-400 mb-2">Your Name</label>
            <input
              type="text"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              placeholder="Enter your name"
              className="w-full bg-gray-700 text-white px-4 py-3 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
            />
          </div>

          <button
            onClick={handleCreateLobby}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-bold mb-3 transition-colors"
          >
            Create New Lobby
          </button>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-700"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-gray-800 text-gray-400">OR</span>
            </div>
          </div>

          <div>
            <label className="block text-gray-400 mb-2">Join Existing Lobby</label>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Lobby ID"
                id="join-lobby-id"
                className="flex-1 bg-gray-700 text-white px-4 py-3 rounded border border-gray-600 focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={() => {
                  const input = document.getElementById('join-lobby-id') as HTMLInputElement;
                  handleJoinLobby(input.value);
                }}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors"
              >
                Join
              </button>
            </div>
          </div>
        </div>

        <div className="mt-6 text-center text-sm text-gray-500">
          <p>Built with Orca Engine</p>
        </div>
      </div>
    </div>
  );
}

// Wrapper component that uses the custom hook
function LobbyWrapper({
  lobbyId,
  playerName,
  onGameStart,
  onLeaveLobby,
}: {
  lobbyId: string;
  playerName: string;
  onGameStart: (mapId: string, settings: any) => void;
  onLeaveLobby: () => void;
}) {
  const {
    connected,
    lobbyState,
    localPlayerId,
    isHost,
    chatMessages,
    error,
    selectMap,
    toggleReady,
    updateTeam,
    updateSettings,
    sendChatMessage,
    startGame,
    leaveLobby,
  } = useGameLobby({
    serverUrl: SERVER_URL,
    lobbyId,
    playerName,
    onGameStart,
  });

  if (!connected) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl mb-4">Connecting to server...</div>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl mb-4 text-red-500">Error</div>
          <p>{error}</p>
          <button
            onClick={onLeaveLobby}
            className="mt-4 bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded"
          >
            Back to Menu
          </button>
        </div>
      </div>
    );
  }

  return (
    <GameLobby
      lobbyId={lobbyId}
      localPlayerId={localPlayerId || ''}
      onStartGame={startGame}
      onLeaveLobby={() => {
        leaveLobby();
        onLeaveLobby();
      }}
    />
  );
}
