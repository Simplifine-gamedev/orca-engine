import React from 'react';
import { useGameStore, PathVisibilityMode } from '../store/gameStore';

export const PathVisibilitySettings: React.FC = () => {
  const {
    pathVisibilityMode,
    showPathLines,
    pathFadeDuration,
    pathOpacity,
    groupDestinationMarkerEnabled,
    setPathVisibilityMode,
    setShowPathLines,
    setPathFadeDuration,
    setPathOpacity,
    setGroupDestinationMarkerEnabled
  } = useGameStore();
  
  const visibilityModes: Array<{ value: PathVisibilityMode; label: string; description: string }> = [
    {
      value: 'lead-only',
      label: 'Lead Unit Only',
      description: 'Show path only for the first selected unit'
    },
    {
      value: 'group-marker',
      label: 'Group Marker',
      description: 'Show single destination marker for entire group'
    },
    {
      value: 'fade-quick',
      label: 'Quick Fade',
      description: 'Show paths that fade quickly'
    },
    {
      value: 'all',
      label: 'All Paths',
      description: 'Show paths for all selected units'
    },
    {
      value: 'none',
      label: 'Hidden',
      description: 'Hide all path lines'
    }
  ];
  
  return (
    <div className="path-visibility-settings">
      <h3 className="settings-title">Path Visibility Settings</h3>
      
      {/* Master toggle */}
      <div className="setting-group">
        <label className="setting-label">
          <input
            type="checkbox"
            checked={showPathLines}
            onChange={(e) => setShowPathLines(e.target.checked)}
          />
          <span>Show Path Lines</span>
        </label>
      </div>
      
      {/* Visibility mode selection */}
      {showPathLines && (
        <>
          <div className="setting-group">
            <label className="setting-label">Path Display Mode</label>
            <div className="mode-selector">
              {visibilityModes.map(mode => (
                <button
                  key={mode.value}
                  className={`mode-button ${pathVisibilityMode === mode.value ? 'active' : ''}`}
                  onClick={() => setPathVisibilityMode(mode.value)}
                  title={mode.description}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>
          
          {/* Group marker toggle */}
          {pathVisibilityMode === 'group-marker' && (
            <div className="setting-group">
              <label className="setting-label">
                <input
                  type="checkbox"
                  checked={groupDestinationMarkerEnabled}
                  onChange={(e) => setGroupDestinationMarkerEnabled(e.target.checked)}
                />
                <span>Show Group Destination Marker</span>
              </label>
            </div>
          )}
          
          {/* Opacity control */}
          {pathVisibilityMode !== 'none' && (
            <div className="setting-group">
              <label className="setting-label">
                Path Opacity: {Math.round(pathOpacity * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={pathOpacity}
                onChange={(e) => setPathOpacity(parseFloat(e.target.value))}
                className="slider"
              />
            </div>
          )}
          
          {/* Fade duration (for fade-quick mode) */}
          {pathVisibilityMode === 'fade-quick' && (
            <div className="setting-group">
              <label className="setting-label">
                Fade Duration: {(pathFadeDuration / 1000).toFixed(1)}s
              </label>
              <input
                type="range"
                min="500"
                max="5000"
                step="100"
                value={pathFadeDuration}
                onChange={(e) => setPathFadeDuration(parseInt(e.target.value))}
                className="slider"
              />
            </div>
          )}
        </>
      )}
      
      <style jsx>{`
        .path-visibility-settings {
          background: rgba(0, 0, 0, 0.8);
          border: 1px solid #4ea7fc;
          border-radius: 8px;
          padding: 16px;
          color: white;
          min-width: 300px;
        }
        
        .settings-title {
          margin: 0 0 16px 0;
          font-size: 18px;
          font-weight: bold;
          color: #4ea7fc;
        }
        
        .setting-group {
          margin-bottom: 16px;
        }
        
        .setting-label {
          display: block;
          margin-bottom: 8px;
          font-size: 14px;
          cursor: pointer;
        }
        
        .setting-label input[type="checkbox"] {
          margin-right: 8px;
          cursor: pointer;
        }
        
        .mode-selector {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .mode-button {
          background: rgba(78, 167, 252, 0.2);
          border: 1px solid #4ea7fc;
          border-radius: 4px;
          padding: 8px 12px;
          color: white;
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        
        .mode-button:hover {
          background: rgba(78, 167, 252, 0.3);
          transform: translateX(4px);
        }
        
        .mode-button.active {
          background: #4ea7fc;
          border-color: #66ccff;
          font-weight: bold;
        }
        
        .slider {
          width: 100%;
          cursor: pointer;
        }
        
        .slider::-webkit-slider-thumb {
          background: #4ea7fc;
        }
        
        .slider::-moz-range-thumb {
          background: #4ea7fc;
        }
      `}</style>
    </div>
  );
};
