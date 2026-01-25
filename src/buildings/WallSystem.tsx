/**
 * Wall System - Manages walls and auto-opening gates for friendly units
 */

import React, { useEffect, useRef, useState } from 'react';
import { wallStore, Gate, Unit, Position } from '../store/wallStore';

interface WallSystemProps {
  detectionRadius?: number; // How close units need to be to trigger gate opening
  closeDelay?: number; // Delay in ms before gate closes after unit passes
  updateInterval?: number; // How often to check for units (ms)
}

export const WallSystem: React.FC<WallSystemProps> = ({
  detectionRadius = 3.0,
  closeDelay = 2000,
  updateInterval = 100,
}) => {
  const [gates, setGates] = useState<Gate[]>([]);
  const [units, setUnits] = useState<Unit[]>([]);
  const animationFrameRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);

  // Subscribe to store changes
  useEffect(() => {
    const updateState = () => {
      setGates(wallStore.getAllGates());
      setUnits(wallStore.getAllUnits());
    };

    updateState();
    const unsubscribe = wallStore.subscribe(updateState);

    return () => {
      unsubscribe();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Main game loop for gate detection
  useEffect(() => {
    const checkGates = (timestamp: number) => {
      // Throttle updates based on updateInterval
      if (timestamp - lastUpdateRef.current >= updateInterval) {
        lastUpdateRef.current = timestamp;
        processGateDetection();
      }

      animationFrameRef.current = requestAnimationFrame(checkGates);
    };

    animationFrameRef.current = requestAnimationFrame(checkGates);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [units, gates, detectionRadius, closeDelay, updateInterval]);

  /**
   * Check all gates for nearby friendly units and handle opening/closing
   */
  const processGateDetection = () => {
    for (const gate of gates) {
      const nearbyFriendlyUnits = getNearbyFriendlyUnits(gate, detectionRadius);

      if (nearbyFriendlyUnits.length > 0) {
        // Friendly units detected - open gate
        if (!gate.isOpen) {
          openGateWithAnimation(gate.id);
        }
        // Keep gate open while units are nearby
        wallStore.scheduleGateClose(gate.id, closeDelay);
      } else {
        // No friendly units nearby - gate will close via scheduled timer
        // (timer is set when last unit was detected)
      }
    }
  };

  /**
   * Get friendly units within detection radius of gate
   */
  const getNearbyFriendlyUnits = (gate: Gate, radius: number): Unit[] => {
    const friendlyUnits: Unit[] = [];

    for (const unit of units) {
      // Check if unit is friendly
      if (!wallStore.isFriendlyUnit(unit.ownerId, gate.ownerId)) {
        continue;
      }

      // Check distance
      const distance = calculateDistance(unit.position, gate.position);
      if (distance <= radius) {
        friendlyUnits.push(unit);
      }
    }

    return friendlyUnits;
  };

  /**
   * Calculate distance between two positions
   */
  const calculateDistance = (pos1: Position, pos2: Position): number => {
    return Math.sqrt(
      Math.pow(pos1.x - pos2.x, 2) +
      Math.pow(pos1.y - pos2.y, 2)
    );
  };

  /**
   * Open gate with animation
   */
  const openGateWithAnimation = (gateId: string) => {
    wallStore.openGate(gateId);
    
    // Trigger animation event (can be extended for actual animation system)
    const event = new CustomEvent('gate-open', {
      detail: { gateId },
    });
    window.dispatchEvent(event);
    
    console.log(`Gate ${gateId} opening for friendly units`);
  };

  /**
   * Close gate with animation
   */
  const closeGateWithAnimation = (gateId: string) => {
    wallStore.closeGate(gateId);
    
    // Trigger animation event
    const event = new CustomEvent('gate-close', {
      detail: { gateId },
    });
    window.dispatchEvent(event);
    
    console.log(`Gate ${gateId} closing`);
  };

  // Render gates and walls (visual representation)
  return (
    <div className="wall-system">
      {gates.map((gate) => (
        <GateComponent
          key={gate.id}
          gate={gate}
          onManualToggle={() => {
            if (gate.isOpen) {
              closeGateWithAnimation(gate.id);
            } else {
              openGateWithAnimation(gate.id);
            }
          }}
        />
      ))}
    </div>
  );
};

/**
 * Individual gate component with visual representation
 */
interface GateComponentProps {
  gate: Gate;
  onManualToggle: () => void;
}

const GateComponent: React.FC<GateComponentProps> = ({ gate, onManualToggle }) => {
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    const handleGateOpen = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail.gateId === gate.id) {
        setIsAnimating(true);
        setTimeout(() => setIsAnimating(false), 500); // Animation duration
      }
    };

    const handleGateClose = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail.gateId === gate.id) {
        setIsAnimating(true);
        setTimeout(() => setIsAnimating(false), 500);
      }
    };

    window.addEventListener('gate-open', handleGateOpen);
    window.addEventListener('gate-close', handleGateClose);

    return () => {
      window.removeEventListener('gate-open', handleGateOpen);
      window.removeEventListener('gate-close', handleGateClose);
    };
  }, [gate.id]);

  return (
    <div
      className={`gate ${gate.isOpen ? 'open' : 'closed'} ${isAnimating ? 'animating' : ''}`}
      style={{
        position: 'absolute',
        left: `${gate.position.x * 32}px`,
        top: `${gate.position.y * 32}px`,
        width: '32px',
        height: '32px',
        backgroundColor: gate.isOpen ? '#4CAF50' : '#795548',
        border: '2px solid #333',
        cursor: 'pointer',
        transition: 'all 0.5s ease',
        transform: gate.isOpen ? 'rotateY(90deg)' : 'rotateY(0deg)',
      }}
      onClick={onManualToggle}
      title={`Gate ${gate.id} - ${gate.isOpen ? 'Open' : 'Closed'}`}
    >
      <div style={{ 
        fontSize: '20px', 
        textAlign: 'center', 
        lineHeight: '28px',
        userSelect: 'none'
      }}>
        {gate.isOpen ? '⬜' : '🚪'}
      </div>
    </div>
  );
};

/**
 * Utility functions for external use
 */
export const WallSystemUtils = {
  /**
   * Create a new gate at position
   */
  createGate: (position: Position, ownerId: string): string => {
    const gateId = `gate_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    wallStore.addGate({
      id: gateId,
      position,
      isOpen: false,
      ownerId,
    });
    return gateId;
  },

  /**
   * Create a new unit
   */
  createUnit: (position: Position, ownerId: string): string => {
    const unitId = `unit_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    wallStore.addUnit({
      id: unitId,
      position,
      ownerId,
    });
    return unitId;
  },

  /**
   * Move unit to new position
   */
  moveUnit: (unitId: string, position: Position): void => {
    wallStore.updateUnitPosition(unitId, position);
  },

  /**
   * Check if gate is currently open
   */
  isGateOpen: (gateId: string): boolean => {
    const gate = wallStore.getGate(gateId);
    return gate?.isOpen ?? false;
  },
};
