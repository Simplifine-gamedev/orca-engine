import { useEffect, useState } from 'react';
import { DamageEvent } from '../types';

interface DamageNumberProps {
  event: DamageEvent;
}

const DamageNumber: React.FC<DamageNumberProps> = ({ event }) => {
  const [opacity, setOpacity] = useState(1);
  const [offsetY, setOffsetY] = useState(0);

  useEffect(() => {
    const startTime = Date.now();
    const duration = 1500; // Animation duration in ms

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Float up
      setOffsetY(-progress * 80);
      
      // Fade out (start fading after 50% of animation)
      if (progress > 0.5) {
        const fadeProgress = (progress - 0.5) * 2;
        setOpacity(1 - fadeProgress);
      }

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }, []);

  // Color based on damage type
  const getColor = () => {
    switch (event.type) {
      case 'critical':
        return '#ff3333'; // Bright red for critical hits
      case 'magic':
        return '#3399ff'; // Blue for magic damage
      case 'physical':
      default:
        return '#ffcc00'; // Yellow for physical damage
    }
  };

  // Size based on damage type
  const getFontSize = () => {
    return event.type === 'critical' ? '32px' : '24px';
  };

  const style: React.CSSProperties = {
    position: 'absolute',
    left: `${event.position.x}px`,
    top: `${event.position.y + offsetY}px`,
    color: getColor(),
    fontSize: getFontSize(),
    fontWeight: 'bold',
    fontFamily: 'Arial, sans-serif',
    opacity: opacity,
    pointerEvents: 'none',
    textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
    transform: 'translate(-50%, -50%)',
    transition: 'none',
    zIndex: 1000,
    userSelect: 'none',
  };

  return <div style={style}>{event.amount}</div>;
};

export default DamageNumber;
