import { useState, useEffect, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

interface Player {
  id: string;
  name: string;
  team: number;
  ready: boolean;
  isHost: boolean;
}

interface LobbyState {
  players: Player[];
  selectedMapId: string | null;
  settings: {
    maxPlayers: number;
    startingResources: string;
    gameSpeed: string;
    fogOfWar: boolean;
  };
}

interface UseGameLobbyOptions {
  serverUrl: string;
  lobbyId: string;
  playerName: string;
  onGameStart?: (mapId: string, settings: any) => void;
}

export function useGameLobby({
  serverUrl,
  lobbyId,
  playerName,
  onGameStart,
}: UseGameLobbyOptions) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lobbyState, setLobbyState] = useState<LobbyState>({
    players: [],
    selectedMapId: null,
    settings: {
      maxPlayers: 4,
      startingResources: 'normal',
      gameSpeed: 'normal',
      fogOfWar: true,
    },
  });
  const [localPlayerId, setLocalPlayerId] = useState<string | null>(null);
  const [isHost, setIsHost] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{
    playerId: string;
    playerName: string;
    message: string;
    timestamp: number;
  }>>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const newSocket = io(serverUrl);
    setSocket(newSocket);

    newSocket.on('connect', () => {
      setConnected(true);
      newSocket.emit('join-lobby', { lobbyId, playerName });
    });

    newSocket.on('disconnect', () => {
      setConnected(false);
    });

    newSocket.on('joined-lobby', ({ playerId, isHost: hostStatus }) => {
      setLocalPlayerId(playerId);
      setIsHost(hostStatus);
    });

    newSocket.on('lobby-update', ({ players, selectedMapId, settings }) => {
      setLobbyState({ players, selectedMapId, settings });
    });

    newSocket.on('map-selected', ({ mapId }) => {
      setLobbyState((prev) => ({ ...prev, selectedMapId: mapId }));
    });

    newSocket.on('settings-updated', ({ settings }) => {
      setLobbyState((prev) => ({ ...prev, settings }));
    });

    newSocket.on('chat-message', (message) => {
      setChatMessages((prev) => [...prev, message]);
    });

    newSocket.on('game-starting', ({ mapId, settings }) => {
      if (onGameStart) {
        onGameStart(mapId, settings);
      }
    });

    newSocket.on('error', ({ message }) => {
      setError(message);
    });

    return () => {
      newSocket.close();
    };
  }, [serverUrl, lobbyId, playerName, onGameStart]);

  const selectMap = useCallback((mapId: string) => {
    if (socket && isHost) {
      socket.emit('select-map', { lobbyId, mapId });
    }
  }, [socket, isHost, lobbyId]);

  const toggleReady = useCallback(() => {
    if (socket) {
      socket.emit('toggle-ready', { lobbyId });
    }
  }, [socket, lobbyId]);

  const updateTeam = useCallback((playerId: string, team: number) => {
    if (socket && isHost) {
      socket.emit('update-team', { lobbyId, playerId, team });
    }
  }, [socket, isHost, lobbyId]);

  const updateSettings = useCallback((settings: any) => {
    if (socket && isHost) {
      socket.emit('update-settings', { lobbyId, settings });
    }
  }, [socket, isHost, lobbyId]);

  const sendChatMessage = useCallback((message: string) => {
    if (socket) {
      socket.emit('chat-message', { lobbyId, message });
    }
  }, [socket, lobbyId]);

  const startGame = useCallback(() => {
    if (socket && isHost) {
      socket.emit('start-game', { lobbyId });
    }
  }, [socket, isHost, lobbyId]);

  const leaveLobby = useCallback(() => {
    if (socket) {
      socket.emit('leave-lobby');
      socket.close();
    }
  }, [socket]);

  return {
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
  };
}
