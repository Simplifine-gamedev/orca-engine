import React, { useEffect, useRef, useState } from 'react';
import { Unit, AnimationState } from '../types/unit';

interface RTSUnitProps {
  unit: Unit;
  onAnimationComplete?: (state: AnimationState) => void;
}

export const RTSUnit: React.FC<RTSUnitProps> = ({ unit, onAnimationComplete }) => {
  const [currentAnimation, setCurrentAnimation] = useState<AnimationState>(unit.animationState);
  const meshRef = useRef<any>(null);
  const mixerRef = useRef<any>(null);
  const actionsRef = useRef<Map<AnimationState, any>>(new Map());

  // Initialize animation mixer and actions
  useEffect(() => {
    if (!meshRef.current) return;

    // Create animation mixer
    // Note: In a real implementation, this would use Three.js AnimationMixer
    // For now, this demonstrates the state machine logic
    
    // FIX: Set initial animation immediately to prevent T-pose
    // This ensures the animation plays from the first frame
    playAnimation(unit.animationState);

    return () => {
      // Cleanup animation mixer
      if (mixerRef.current) {
        mixerRef.current.stopAllAction();
      }
    };
  }, []);

  // Update animation when unit state changes
  useEffect(() => {
    if (currentAnimation !== unit.animationState) {
      transitionToAnimation(unit.animationState);
    }
  }, [unit.animationState]);

  const playAnimation = (state: AnimationState) => {
    // FIX: Animation state machine with proper initialization
    // This prevents T-pose by ensuring an animation is always active
    
    if (!meshRef.current) return;

    // Stop current animation
    const currentAction = actionsRef.current.get(currentAnimation);
    if (currentAction) {
      currentAction.fadeOut(0.2);
    }

    // Get or create new animation action
    let newAction = actionsRef.current.get(state);
    
    // Animation clips would be loaded from the model
    // For demonstration, showing the state machine logic
    const animationClips: Record<AnimationState, any> = {
      spawning: { duration: 0.5, loop: false },
      idle: { duration: 2.0, loop: true },
      walking: { duration: 1.0, loop: true },
      attacking: { duration: 0.8, loop: false },
      dying: { duration: 1.2, loop: false },
    };

    const clip = animationClips[state];
    
    if (!newAction && clip) {
      // In real implementation: newAction = mixer.clipAction(clip)
      // For now, simulating the action
      newAction = {
        fadeIn: (duration: number) => {},
        play: () => {},
        fadeOut: (duration: number) => {},
        setLoop: (loop: boolean) => {},
        duration: clip.duration,
      };
      actionsRef.current.set(state, newAction);
    }

    if (newAction) {
      // FIX: Play animation immediately with no delay
      // This prevents showing the default T-pose
      newAction.fadeIn(0.2);
      newAction.play();
      newAction.setLoop(clip.loop);
      
      setCurrentAnimation(state);

      // Handle animation completion for non-looping animations
      if (!clip.loop) {
        setTimeout(() => {
          if (onAnimationComplete) {
            onAnimationComplete(state);
          }
          
          // Auto-transition to appropriate next state
          if (state === 'spawning') {
            playAnimation('idle');
          } else if (state === 'attacking') {
            playAnimation('idle');
          } else if (state === 'dying') {
            // Unit should be removed by game logic
          }
        }, clip.duration * 1000);
      }
    }
  };

  const transitionToAnimation = (newState: AnimationState) => {
    playAnimation(newState);
  };

  return (
    <div
      ref={meshRef}
      style={{
        position: 'absolute',
        left: `${unit.position.x}px`,
        top: `${unit.position.y}px`,
        transition: 'left 0.3s, top 0.3s',
      }}
      className="rts-unit"
      data-animation={currentAnimation}
      data-unit-id={unit.id}
    >
      {/* Visual representation of the unit */}
      {/* In a real implementation, this would render a 3D model */}
      <div className="unit-sprite">
        <div className="health-bar">
          <div
            className="health-fill"
            style={{
              width: `${(unit.health / unit.maxHealth) * 100}%`,
            }}
          />
        </div>
        <div className="animation-state">{currentAnimation}</div>
      </div>
    </div>
  );
};

export default RTSUnit;
