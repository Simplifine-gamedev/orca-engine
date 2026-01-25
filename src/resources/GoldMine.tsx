import React from 'react';

interface GoldMineProps {
  remainingGold: number;
  maxGold: number;
  position: { x: number; y: number };
  isSelected?: boolean;
}

export const GoldMine: React.FC<GoldMineProps> = ({
  remainingGold,
  maxGold,
  position,
  isSelected = false
}) => {
  const depletionPercentage = (remainingGold / maxGold) * 100;
  const isLow = depletionPercentage < 30;
  const isCritical = depletionPercentage < 10;

  return (
    <div
      className="gold-mine"
      style={{
        position: 'absolute',
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      {/* Gold Mine Sprite */}
      <div className="mine-sprite">
        <div className="mine-icon">⛏️</div>
        
        {/* Improved Depletion Indicator - Much Larger and More Visible */}
        <div className="depletion-display">
          {/* Large Number Display */}
          <div 
            className={`gold-amount ${isLow ? 'low' : ''} ${isCritical ? 'critical' : ''}`}
            style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: isCritical ? '#ff4444' : isLow ? '#ffaa00' : '#ffd700',
              textShadow: '2px 2px 4px rgba(0,0,0,0.8), -1px -1px 2px rgba(0,0,0,0.8)',
              backgroundColor: 'rgba(0,0,0,0.7)',
              padding: '6px 12px',
              borderRadius: '8px',
              border: '2px solid rgba(255,215,0,0.6)',
              marginBottom: '4px',
            }}
          >
            {remainingGold.toLocaleString()}
          </div>

          {/* Progress Bar */}
          <div 
            className="gold-progress-bar"
            style={{
              width: '80px',
              height: '12px',
              backgroundColor: 'rgba(0,0,0,0.7)',
              borderRadius: '6px',
              border: '2px solid rgba(255,215,0,0.4)',
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            <div 
              className="progress-fill"
              style={{
                width: `${depletionPercentage}%`,
                height: '100%',
                backgroundColor: isCritical ? '#ff4444' : isLow ? '#ffaa00' : '#ffd700',
                transition: 'width 0.3s ease, background-color 0.3s ease',
                boxShadow: 'inset 0 2px 4px rgba(255,255,255,0.3)',
              }}
            />
            {/* Percentage Text on Bar */}
            <div 
              style={{
                position: 'absolute',
                top: '0',
                left: '0',
                right: '0',
                bottom: '0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                fontWeight: 'bold',
                color: '#fff',
                textShadow: '1px 1px 2px rgba(0,0,0,0.8)',
              }}
            >
              {Math.round(depletionPercentage)}%
            </div>
          </div>
        </div>
      </div>

      {/* Selection Panel - Shows when mine is selected */}
      {isSelected && (
        <div className="mine-selection-panel">
          <GoldMineSelectionPanel 
            remainingGold={remainingGold}
            maxGold={maxGold}
            depletionPercentage={depletionPercentage}
          />
        </div>
      )}

      <style jsx>{`
        .gold-mine {
          user-select: none;
          cursor: pointer;
        }

        .mine-sprite {
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .mine-icon {
          font-size: 48px;
          filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.5));
        }

        .depletion-display {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
          pointer-events: none;
        }

        .gold-amount.critical {
          animation: pulse-critical 1s infinite;
        }

        .gold-amount.low {
          animation: pulse-low 2s infinite;
        }

        @keyframes pulse-critical {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }

        @keyframes pulse-low {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.8; }
        }

        .mine-selection-panel {
          position: fixed;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          z-index: 1000;
        }
      `}</style>
    </div>
  );
};

interface SelectionPanelProps {
  remainingGold: number;
  maxGold: number;
  depletionPercentage: number;
}

export const GoldMineSelectionPanel: React.FC<SelectionPanelProps> = ({
  remainingGold,
  maxGold,
  depletionPercentage
}) => {
  const isLow = depletionPercentage < 30;
  const isCritical = depletionPercentage < 10;

  return (
    <div 
      className="selection-panel"
      style={{
        backgroundColor: 'rgba(20, 20, 30, 0.95)',
        border: '3px solid #ffd700',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '300px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
      }}
    >
      <h3 
        style={{
          margin: '0 0 16px 0',
          fontSize: '20px',
          color: '#ffd700',
          textAlign: 'center',
          borderBottom: '2px solid #ffd700',
          paddingBottom: '8px',
        }}
      >
        ⛏️ Gold Mine
      </h3>

      {/* Large Gold Amount Display */}
      <div 
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '12px',
        }}
      >
        <span style={{ fontSize: '16px', color: '#ddd' }}>Remaining Gold:</span>
        <span 
          style={{
            fontSize: '28px',
            fontWeight: 'bold',
            color: isCritical ? '#ff4444' : isLow ? '#ffaa00' : '#ffd700',
            textShadow: '2px 2px 4px rgba(0,0,0,0.8)',
          }}
        >
          {remainingGold.toLocaleString()}
        </span>
      </div>

      {/* Max Gold */}
      <div 
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '12px',
          fontSize: '14px',
          color: '#aaa',
        }}
      >
        <span>Maximum:</span>
        <span>{maxGold.toLocaleString()}</span>
      </div>

      {/* Large Progress Bar */}
      <div style={{ marginBottom: '12px' }}>
        <div 
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: '6px',
            fontSize: '14px',
            color: '#ddd',
          }}
        >
          <span>Depletion</span>
          <span 
            style={{
              fontWeight: 'bold',
              color: isCritical ? '#ff4444' : isLow ? '#ffaa00' : '#44ff44',
            }}
          >
            {Math.round(depletionPercentage)}%
          </span>
        </div>
        <div 
          style={{
            width: '100%',
            height: '24px',
            backgroundColor: 'rgba(0,0,0,0.6)',
            borderRadius: '12px',
            border: '2px solid rgba(255,215,0,0.5)',
            overflow: 'hidden',
            position: 'relative',
          }}
        >
          <div 
            style={{
              width: `${depletionPercentage}%`,
              height: '100%',
              background: isCritical 
                ? 'linear-gradient(90deg, #ff4444, #ff6666)' 
                : isLow 
                  ? 'linear-gradient(90deg, #ffaa00, #ffcc44)'
                  : 'linear-gradient(90deg, #ffd700, #ffed4e)',
              transition: 'width 0.3s ease',
              boxShadow: 'inset 0 2px 6px rgba(255,255,255,0.3)',
            }}
          />
        </div>
      </div>

      {/* Warning Messages */}
      {isCritical && (
        <div 
          style={{
            backgroundColor: 'rgba(255,68,68,0.2)',
            border: '2px solid #ff4444',
            borderRadius: '6px',
            padding: '10px',
            marginTop: '12px',
            fontSize: '14px',
            color: '#ff4444',
            fontWeight: 'bold',
            textAlign: 'center',
            animation: 'pulse 1s infinite',
          }}
        >
          ⚠️ CRITICAL: Mine Nearly Depleted!
        </div>
      )}
      {isLow && !isCritical && (
        <div 
          style={{
            backgroundColor: 'rgba(255,170,0,0.2)',
            border: '2px solid #ffaa00',
            borderRadius: '6px',
            padding: '10px',
            marginTop: '12px',
            fontSize: '14px',
            color: '#ffaa00',
            fontWeight: 'bold',
            textAlign: 'center',
          }}
        >
          ⚠️ Warning: Gold Running Low
        </div>
      )}

      {/* Info Text */}
      <div 
        style={{
          marginTop: '16px',
          fontSize: '12px',
          color: '#888',
          textAlign: 'center',
          fontStyle: 'italic',
        }}
      >
        Workers automatically gather gold from this mine
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  );
};

export default GoldMine;
