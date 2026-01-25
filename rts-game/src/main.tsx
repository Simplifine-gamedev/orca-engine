import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { damageNumberStyles } from './effects/DamageNumber';

// Inject damage number styles
const styleElement = document.createElement('style');
styleElement.textContent = `
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    margin: 0;
    padding: 0;
    overflow-x: hidden;
  }

  ${damageNumberStyles}
`;
document.head.appendChild(styleElement);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
