import { useEffect, useState } from 'react';
import { DamageEvent, DamageType } from '../types';

interface DamageNumberProps {
  event: DamageEvent;
  onComplete: (id: string) => void;
}

// Color mapping for different damage types
const DAMAGE_COLORS: Record<DamageType, string> = {
  physical: '#FF6B6B',      // Red
  magical: '#A78BFA',       // Purple
  fire: '#F97316',          // Orange
  ice: '#60A5FA',           // Light Blue
  poison: '#84CC16',        // Lime Green
  healing: '#4ADE80',       // Green
};

export const DamageNumber: React.FC<DamageNumberProps> = ({ event, onComplete }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    // Animation duration: 1.5 seconds
    const timer = setTimeout(() => {
      setIsVisible(false);
      onComplete(event.id);
    }, 1500);

    return () => clearTimeout(timer);
  }, [event.id, onComplete]);

  if (!isVisible) return null;

  const color = DAMAGE_COLORS[event.type];
  const isHealing = event.type === 'healing';
  const displayAmount = isHealing ? `+${event.amount}` : `-${event.amount}`;

  return (
    <div
      className="damage-number"
      style={{
        position: 'absolute',
        left: `${event.x}px`,
        top: `${event.y}px`,
        color: color,
        fontSize: '24px',
        fontWeight: 'bold',
        fontFamily: 'monospace',
        pointerEvents: 'none',
        userSelect: 'none',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
        animation: 'floatUp 1.5s ease-out forwards',
        zIndex: 1000,
      }}
    >
      {displayAmount}
    </div>
  );
};

// CSS animation styles
export const damageNumberStyles = `
  @keyframes floatUp {
    0% {
      transform: translateY(0) scale(1);
      opacity: 1;
    }
    50% {
      transform: translateY(-30px) scale(1.2);
      opacity: 1;
    }
    100% {
      transform: translateY(-60px) scale(0.8);
      opacity: 0;
    }
  }

  .damage-number {
    will-change: transform, opacity;
  }
`;
