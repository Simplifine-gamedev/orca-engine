/**
 * GameLobby Component
 * Main lobby interface for Orca RTS with map selection
 */

import React, { useState, useEffect } from 'react';
import WorldMapSelector from './WorldMapSelector';
import { MapPreset } from '../types/MapTypes';
import { getMapById } from '../config/mapPresets';

interface Player {
  id: string;
  name: string;
  isHost: boolean;
  isReady: boolean;
  team?: number;
  color: string;
}

interface GameLobbyProps {
  playerId: string;
  playerName: string;
  isHost: boolean;
  onStartGame: (mapId: string) => void;
  onLeaveLobby: () => void;
}

export const GameLobby: React.FC<GameLobbyProps> = ({
  playerId,
  playerName,
  isHost,
  onStartGame,
  onLeaveLobby,
}) => {
  const [players, setPlayers] = useState<Player[]>([
    {
      id: playerId,
      name: playerName,
      isHost: isHost,
      isReady: false,
      color: '#3B82F6',
    },
  ]);
  const [selectedMapId, setSelectedMapId] = useState<string>('medium-pangaea');
  const [showMapSelector, setShowMapSelector] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<Array<{ player: string; message: string }>>([]);
  const [chatInput, setChatInput] = useState<string>('');

  const selectedMap = getMapById(selectedMapId);
  const allPlayersReady = players.every((p) => p.isReady || p.isHost);
  const canStartGame = isHost && allPlayersReady && players.length >= 2;

  const handleMapSelect = (map: MapPreset) => {
    setSelectedMapId(map.id);
    setShowMapSelector(false);
    
    // Notify server about map selection
    // In a real implementation, this would send to the server
    console.log('Map selected:', map.id);
  };

  const handleToggleReady = () => {
    setPlayers((prev) =>
      prev.map((p) =>
        p.id === playerId ? { ...p, isReady: !p.isReady } : p
      )
    );
  };

  const handleStartGame = () => {
    if (canStartGame) {
      onStartGame(selectedMapId);
    }
  };

  const handleSendMessage = () => {
    if (chatInput.trim()) {
      setChatMessages((prev) => [
        ...prev,
        { player: playerName, message: chatInput },
      ]);
      setChatInput('');
    }
  };

  return (
    <div
      className="game-lobby"
      style={{
        display: 'flex',
        height: '100vh',
        backgroundColor: '#0a0a0a',
        color: 'white',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      {/* Left Panel - Players and Settings */}
      <div
        style={{
          width: '350px',
          backgroundColor: '#1a1a1a',
          padding: '20px',
          borderRight: '1px solid #333',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Lobby Header */}
        <div style={{ marginBottom: '20px' }}>
          <h1 style={{ fontSize: '24px', margin: '0 0 10px 0' }}>Game Lobby</h1>
          <div style={{ fontSize: '12px', color: '#888' }}>
            Orca RTS - Multiplayer Match
          </div>
        </div>

        {/* Map Selection */}
        <div
          style={{
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: '#252525',
            borderRadius: '8px',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '10px',
            }}
          >
            <h3 style={{ margin: 0, fontSize: '16px' }}>Selected Map</h3>
            {isHost && (
              <button
                onClick={() => setShowMapSelector(!showMapSelector)}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#3B82F6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px',
                }}
              >
                Change Map
              </button>
            )}
          </div>

          {selectedMap && (
            <div style={{ fontSize: '14px' }}>
              <div
                style={{
                  padding: '10px',
                  backgroundColor: selectedMap.previewColor,
                  borderRadius: '4px',
                  marginBottom: '10px',
                  height: '80px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '18px',
                }}
              >
                {selectedMap.name}
              </div>
              <div style={{ fontSize: '12px', color: '#aaa' }}>
                <div>Size: {selectedMap.size.toUpperCase()}</div>
                <div>Dimensions: {selectedMap.width}x{selectedMap.height}</div>
                <div>Max Players: {selectedMap.maxPlayers}</div>
                <div>Layout: {selectedMap.layout}</div>
              </div>
            </div>
          )}
        </div>

        {/* Players List */}
        <div style={{ flex: 1, overflow: 'auto', marginBottom: '20px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '10px' }}>
            Players ({players.length}/{selectedMap?.maxPlayers || 8})
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {players.map((player) => (
              <div
                key={player.id}
                style={{
                  padding: '12px',
                  backgroundColor: '#252525',
                  borderRadius: '6px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                }}
              >
                <div
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    backgroundColor: player.color,
                  }}
                />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 'bold' }}>
                    {player.name}
                    {player.isHost && (
                      <span
                        style={{
                          marginLeft: '6px',
                          fontSize: '10px',
                          padding: '2px 6px',
                          backgroundColor: '#F59E0B',
                          borderRadius: '3px',
                        }}
                      >
                        HOST
                      </span>
                    )}
                  </div>
                </div>
                <div
                  style={{
                    fontSize: '11px',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    backgroundColor: player.isReady ? '#166534' : '#7C2D12',
                    fontWeight: 'bold',
                  }}
                >
                  {player.isReady ? 'READY' : 'NOT READY'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {!isHost && (
            <button
              onClick={handleToggleReady}
              style={{
                padding: '12px',
                backgroundColor: players.find((p) => p.id === playerId)?.isReady
                  ? '#DC2626'
                  : '#16A34A',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold',
              }}
            >
              {players.find((p) => p.id === playerId)?.isReady
                ? 'Not Ready'
                : 'Ready'}
            </button>
          )}

          {isHost && (
            <button
              onClick={handleStartGame}
              disabled={!canStartGame}
              style={{
                padding: '12px',
                backgroundColor: canStartGame ? '#16A34A' : '#555',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: canStartGame ? 'pointer' : 'not-allowed',
                fontSize: '14px',
                fontWeight: 'bold',
              }}
            >
              Start Game
            </button>
          )}

          <button
            onClick={onLeaveLobby}
            style={{
              padding: '12px',
              backgroundColor: '#DC2626',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
            }}
          >
            Leave Lobby
          </button>
        </div>
      </div>

      {/* Right Panel - Map Selector or Chat */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {showMapSelector ? (
          <div style={{ flex: 1, overflow: 'auto', backgroundColor: '#0a0a0a' }}>
            <WorldMapSelector
              onMapSelect={handleMapSelect}
              selectedMapId={selectedMapId}
              maxPlayers={players.length}
            />
          </div>
        ) : (
          <>
            {/* Map Preview */}
            <div
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#0a0a0a',
                position: 'relative',
              }}
            >
              {selectedMap && (
                <div style={{ textAlign: 'center', maxWidth: '600px' }}>
                  <div
                    style={{
                      width: '500px',
                      height: '500px',
                      backgroundColor: selectedMap.previewColor,
                      borderRadius: '12px',
                      margin: '0 auto 20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '72px',
                      fontWeight: 'bold',
                      opacity: 0.8,
                      position: 'relative',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Water overlay */}
                    <div
                      style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: `${selectedMap.terrain.water}%`,
                        backgroundColor: 'rgba(59, 130, 246, 0.4)',
                      }}
                    />
                    <div style={{ position: 'relative', zIndex: 1 }}>
                      {selectedMap.name}
                    </div>
                  </div>
                  <p style={{ fontSize: '18px', color: '#aaa' }}>
                    {selectedMap.description}
                  </p>
                </div>
              )}
            </div>

            {/* Chat */}
            <div
              style={{
                height: '200px',
                backgroundColor: '#1a1a1a',
                borderTop: '1px solid #333',
                display: 'flex',
                flexDirection: 'column',
                padding: '15px',
              }}
            >
              <h3 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>Chat</h3>
              <div
                style={{
                  flex: 1,
                  overflow: 'auto',
                  marginBottom: '10px',
                  fontSize: '13px',
                }}
              >
                {chatMessages.map((msg, idx) => (
                  <div key={idx} style={{ marginBottom: '6px' }}>
                    <span style={{ color: '#3B82F6', fontWeight: 'bold' }}>
                      {msg.player}:
                    </span>{' '}
                    {msg.message}
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Type a message..."
                  style={{
                    flex: 1,
                    padding: '8px',
                    backgroundColor: '#252525',
                    border: '1px solid #444',
                    borderRadius: '4px',
                    color: 'white',
                    fontSize: '13px',
                  }}
                />
                <button
                  onClick={handleSendMessage}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#3B82F6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '13px',
                  }}
                >
                  Send
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default GameLobby;
