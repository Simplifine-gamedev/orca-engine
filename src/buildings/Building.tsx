import React, { useState, useEffect } from 'react';
import { buildingModels } from './buildingModels';

export interface BuildingProps {
  id: string;
  type: string;
  position: { x: number; y: number };
  faction: string;
  isPreview?: boolean;
  onSelect?: (id: string) => void;
  onComplete?: (id: string) => void;
}

export interface BuildingState {
  health: number;
  maxHealth: number;
  isConstructed: boolean;
  constructionProgress: number;
  trainingQueue: string[];
  currentTraining?: {
    unitType: string;
    progress: number;
  };
}

export const Building: React.FC<BuildingProps> = ({
  id,
  type,
  position,
  faction,
  isPreview = false,
  onSelect,
  onComplete,
}) => {
  const [state, setState] = useState<BuildingState>({
    health: 100,
    maxHealth: 100,
    isConstructed: !isPreview,
    constructionProgress: isPreview ? 0 : 100,
    trainingQueue: [],
  });

  const buildingConfig = buildingModels[type];

  useEffect(() => {
    if (!state.isConstructed && !isPreview) {
      const interval = setInterval(() => {
        setState((prev) => {
          const newProgress = Math.min(prev.constructionProgress + 1, 100);
          const isNowComplete = newProgress >= 100;
          
          if (isNowComplete && !prev.isConstructed) {
            onComplete?.(id);
          }

          return {
            ...prev,
            constructionProgress: newProgress,
            isConstructed: isNowComplete,
          };
        });
      }, 100);

      return () => clearInterval(interval);
    }
  }, [state.isConstructed, isPreview, id, onComplete]);

  useEffect(() => {
    if (
      state.isConstructed &&
      state.trainingQueue.length > 0 &&
      !state.currentTraining
    ) {
      const unitType = state.trainingQueue[0];
      setState((prev) => ({
        ...prev,
        currentTraining: { unitType, progress: 0 },
        trainingQueue: prev.trainingQueue.slice(1),
      }));
    }

    if (state.currentTraining) {
      const interval = setInterval(() => {
        setState((prev) => {
          if (!prev.currentTraining) return prev;

          const newProgress = prev.currentTraining.progress + 1;
          
          if (newProgress >= 100) {
            return {
              ...prev,
              currentTraining: undefined,
            };
          }

          return {
            ...prev,
            currentTraining: {
              ...prev.currentTraining,
              progress: newProgress,
            },
          };
        });
      }, 50);

      return () => clearInterval(interval);
    }
  }, [state.isConstructed, state.trainingQueue, state.currentTraining]);

  const handleClick = () => {
    if (!isPreview && onSelect) {
      onSelect(id);
    }
  };

  const addToTrainingQueue = (unitType: string) => {
    setState((prev) => ({
      ...prev,
      trainingQueue: [...prev.trainingQueue, unitType],
    }));
  };

  if (!buildingConfig) {
    console.error(`Building type "${type}" not found in buildingModels`);
    return null;
  }

  const opacity = isPreview ? 0.5 : 1;
  const cursor = isPreview ? 'not-allowed' : 'pointer';

  return (
    <div
      className="building"
      onClick={handleClick}
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        width: buildingConfig.width || 64,
        height: buildingConfig.height || 64,
        opacity,
        cursor,
        border: '2px solid #333',
        backgroundColor: buildingConfig.color || '#8b4513',
        borderRadius: '4px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: isPreview ? 'none' : '0 2px 4px rgba(0,0,0,0.3)',
      }}
    >
      {buildingConfig.model ? (
        <div
          className="building-model"
          style={{
            width: '100%',
            height: '100%',
            backgroundImage: `url(${buildingConfig.model})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
      ) : (
        <div className="building-placeholder">
          <span style={{ fontSize: '12px', color: '#fff', textAlign: 'center' }}>
            {buildingConfig.name || type}
          </span>
        </div>
      )}

      {!state.isConstructed && !isPreview && (
        <div
          className="construction-progress"
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: '8px',
            backgroundColor: 'rgba(0,0,0,0.5)',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${state.constructionProgress}%`,
              backgroundColor: '#4caf50',
              transition: 'width 0.1s',
            }}
          />
        </div>
      )}

      {state.currentTraining && (
        <div
          className="training-progress"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '4px',
            backgroundColor: 'rgba(0,0,0,0.5)',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${state.currentTraining.progress}%`,
              backgroundColor: '#2196f3',
              transition: 'width 0.05s',
            }}
          />
        </div>
      )}

      {!isPreview && state.trainingQueue.length > 0 && (
        <div
          className="training-queue-indicator"
          style={{
            position: 'absolute',
            top: '4px',
            right: '4px',
            backgroundColor: 'rgba(33, 150, 243, 0.8)',
            borderRadius: '50%',
            width: '16px',
            height: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '10px',
            color: '#fff',
          }}
        >
          {state.trainingQueue.length}
        </div>
      )}
    </div>
  );
};

export default Building;
