import { useEffect } from 'react';
import GameCanvas from './components/GameCanvas';
import UI from './components/UI';
import useGameStore from './store/gameStore';
import './App.css';

function App() {
  const { initializeGame, toggleDamageNumbers } = useGameStore();

  useEffect(() => {
    // Initialize the game on mount
    initializeGame();

    // Add keyboard listener for space key
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        toggleDamageNumbers();
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [initializeGame, toggleDamageNumbers]);

  return (
    <div className="app">
      <GameCanvas />
      <UI />
    </div>
  );
}

export default App;
