import React, { useEffect, useState } from 'react';
import { LoadingProgress } from '../systems/AssetPreloader';

interface LoadingOverlayProps {
  progress: LoadingProgress;
  isLoading: boolean;
  error?: string;
}

/**
 * LoadingOverlay - Shows loading progress for each asset type
 * Fixes the issue where models load during gameplay making it feel buggy/laggy
 * 
 * Features:
 * - Shows overall loading percentage
 * - Displays current asset being loaded
 * - Shows asset type (model, texture, audio)
 * - Displays error messages if loading fails
 * - Only allows game to start when all critical assets are loaded
 */
export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  progress,
  isLoading,
  error,
}) => {
  const [showOverlay, setShowOverlay] = useState(true);

  useEffect(() => {
    // Keep overlay visible until loading is complete and a brief delay
    if (!isLoading && progress.percentage === 100) {
      const timer = setTimeout(() => {
        setShowOverlay(false);
      }, 500); // Brief delay to show 100% before hiding

      return () => clearTimeout(timer);
    }
  }, [isLoading, progress.percentage]);

  if (!showOverlay) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.95)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 9999,
        fontFamily: 'Arial, sans-serif',
        color: '#ffffff',
      }}
    >
      {/* Logo/Title */}
      <div
        style={{
          fontSize: '48px',
          fontWeight: 'bold',
          marginBottom: '40px',
          textAlign: 'center',
        }}
      >
        Orca RTS
      </div>

      {/* Error Display */}
      {error && (
        <div
          style={{
            backgroundColor: '#ff4444',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '20px',
            maxWidth: '600px',
            textAlign: 'center',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
            Loading Error
          </div>
          <div>{error}</div>
        </div>
      )}

      {/* Loading Animation */}
      {isLoading && !error && (
        <>
          {/* Progress Bar Container */}
          <div
            style={{
              width: '600px',
              height: '40px',
              backgroundColor: '#333333',
              borderRadius: '20px',
              overflow: 'hidden',
              marginBottom: '20px',
              border: '2px solid #555555',
            }}
          >
            {/* Progress Bar Fill */}
            <div
              style={{
                height: '100%',
                width: `${progress.percentage}%`,
                backgroundColor: '#4CAF50',
                transition: 'width 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold',
                fontSize: '18px',
              }}
            >
              {progress.percentage > 10 && `${progress.percentage}%`}
            </div>
          </div>

          {/* Loading Stats */}
          <div
            style={{
              textAlign: 'center',
              fontSize: '18px',
              marginBottom: '10px',
            }}
          >
            Loading Assets: {progress.loaded} / {progress.total}
          </div>

          {/* Current Asset Info */}
          {progress.currentAsset && (
            <div
              style={{
                textAlign: 'center',
                fontSize: '14px',
                color: '#aaaaaa',
                marginBottom: '10px',
              }}
            >
              <div style={{ marginBottom: '5px' }}>
                <span
                  style={{
                    display: 'inline-block',
                    padding: '4px 12px',
                    backgroundColor: getAssetTypeColor(progress.assetType),
                    borderRadius: '12px',
                    fontSize: '12px',
                    textTransform: 'uppercase',
                    fontWeight: 'bold',
                    marginRight: '8px',
                  }}
                >
                  {progress.assetType || 'loading'}
                </span>
              </div>
              <div>{progress.currentAsset}</div>
            </div>
          )}

          {/* Loading Spinner */}
          <div
            style={{
              marginTop: '20px',
              width: '40px',
              height: '40px',
              border: '4px solid #333333',
              borderTop: '4px solid #4CAF50',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }}
          />
        </>
      )}

      {/* Completion Message */}
      {!isLoading && progress.percentage === 100 && !error && (
        <div
          style={{
            fontSize: '24px',
            fontWeight: 'bold',
            color: '#4CAF50',
            marginTop: '20px',
          }}
        >
          Ready! Starting game...
        </div>
      )}

      {/* CSS Animation */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

/**
 * Get color based on asset type
 */
function getAssetTypeColor(assetType: string | null): string {
  switch (assetType) {
    case 'model':
      return '#2196F3'; // Blue for models
    case 'texture':
      return '#FF9800'; // Orange for textures
    case 'audio':
      return '#9C27B0'; // Purple for audio
    default:
      return '#666666'; // Gray for unknown
  }
}
