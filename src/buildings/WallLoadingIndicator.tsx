import React from 'react';

interface WallLoadingIndicatorProps {
  isLoading: boolean;
  message?: string;
}

/**
 * Loading indicator shown while wall preview assets are being loaded
 * Displays a simple progress indicator to inform users that the system is preparing
 */
export const WallLoadingIndicator: React.FC<WallLoadingIndicatorProps> = ({
  isLoading,
  message = 'Loading wall blueprints...',
}) => {
  if (!isLoading) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        padding: '20px 40px',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        borderRadius: '8px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
        zIndex: 1000,
        fontFamily: 'Arial, sans-serif',
      }}
    >
      <div
        style={{
          width: '40px',
          height: '40px',
          border: '4px solid rgba(255, 255, 255, 0.3)',
          borderTop: '4px solid white',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }}
      />
      <span style={{ fontSize: '16px' }}>{message}</span>
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default WallLoadingIndicator;
