import React from 'react';
import ReactDOM from 'react-dom/client';
import { Game } from './components/Game';

const root = document.getElementById('root');

if (root) {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <Game />
    </React.StrictMode>
  );
}
