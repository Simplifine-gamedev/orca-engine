import React from 'react';

interface GoldMineProps {
  goldRemaining: number;
  maxGold: number;
  position: { x: number; y: number };
  isSelected?: boolean;
}

export const GoldMine: React.FC<GoldMineProps> = ({
  goldRemaining,
  maxGold,
  position,
  isSelected = false,
}) => {
  const depletionPercentage = (goldRemaining / maxGold) * 100;
  const isNearlyDepleted = depletionPercentage < 25;
  const isDepleted = goldRemaining <= 0;

  return (
    <div
      className="gold-mine"
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
      }}
    >
      {/* Mine visual representation */}
      <div className="mine-sprite">
        <img
          src="/assets/gold-mine.png"
          alt="Gold Mine"
          className={isDepleted ? 'depleted' : ''}
        />
      </div>

      {/* Enhanced depletion indicator - always visible */}
      <div className="depletion-indicator">
        {/* Large, readable text */}
        <div
          className="gold-amount-text"
          style={{
            fontSize: '18px',
            fontWeight: 'bold',
            color: isNearlyDepleted ? '#ff4444' : '#ffd700',
            textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
            marginBottom: '4px',
            textAlign: 'center',
          }}
        >
          {goldRemaining} / {maxGold}
        </div>

        {/* Progress bar for visual feedback */}
        <div
          className="gold-progress-bar"
          style={{
            width: '80px',
            height: '12px',
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            borderRadius: '6px',
            overflow: 'hidden',
            border: '2px solid #333',
          }}
        >
          <div
            className="progress-fill"
            style={{
              width: `${depletionPercentage}%`,
              height: '100%',
              backgroundColor: isNearlyDepleted ? '#ff4444' : '#ffd700',
              transition: 'width 0.3s ease, background-color 0.3s ease',
              boxShadow: 'inset 0 2px 4px rgba(255, 255, 255, 0.3)',
            }}
          />
        </div>
      </div>

      {/* Selection panel enhancement */}
      {isSelected && (
        <div className="selection-panel-info">
          <GoldMineSelectionPanel
            goldRemaining={goldRemaining}
            maxGold={maxGold}
            depletionPercentage={depletionPercentage}
          />
        </div>
      )}

      <style jsx>{`
        .gold-mine {
          user-select: none;
        }

        .mine-sprite {
          position: relative;
        }

        .mine-sprite img {
          width: 64px;
          height: 64px;
        }

        .mine-sprite img.depleted {
          opacity: 0.5;
          filter: grayscale(0.7);
        }

        .depletion-indicator {
          position: absolute;
          bottom: -35px;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          flex-direction: column;
          align-items: center;
          pointer-events: none;
        }

        .gold-mine:hover .depletion-indicator {
          transform: translateX(-50%) scale(1.1);
        }
      `}</style>
    </div>
  );
};

interface SelectionPanelProps {
  goldRemaining: number;
  maxGold: number;
  depletionPercentage: number;
}

const GoldMineSelectionPanel: React.FC<SelectionPanelProps> = ({
  goldRemaining,
  maxGold,
  depletionPercentage,
}) => {
  return (
    <div
      className="gold-mine-selection-details"
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        backgroundColor: 'rgba(20, 20, 30, 0.95)',
        border: '3px solid #ffd700',
        borderRadius: '8px',
        padding: '20px',
        minWidth: '280px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
      }}
    >
      <h3
        style={{
          margin: '0 0 15px 0',
          fontSize: '20px',
          color: '#ffd700',
          borderBottom: '2px solid #ffd700',
          paddingBottom: '8px',
        }}
      >
        Gold Mine
      </h3>

      <div className="stat-row" style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '14px', color: '#aaa', marginBottom: '4px' }}>
          Gold Remaining:
        </div>
        <div
          style={{
            fontSize: '28px',
            fontWeight: 'bold',
            color: '#ffd700',
            textShadow: '0 2px 4px rgba(0, 0, 0, 0.8)',
          }}
        >
          {goldRemaining.toLocaleString()}
        </div>
      </div>

      <div className="stat-row" style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '14px', color: '#aaa', marginBottom: '4px' }}>
          Maximum Capacity:
        </div>
        <div style={{ fontSize: '16px', color: '#ccc' }}>
          {maxGold.toLocaleString()}
        </div>
      </div>

      <div className="stat-row" style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '14px', color: '#aaa', marginBottom: '8px' }}>
          Depletion Status:
        </div>
        <div
          style={{
            width: '100%',
            height: '24px',
            backgroundColor: '#222',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '2px solid #444',
            position: 'relative',
          }}
        >
          <div
            style={{
              width: `${depletionPercentage}%`,
              height: '100%',
              backgroundColor:
                depletionPercentage < 25
                  ? '#ff4444'
                  : depletionPercentage < 50
                  ? '#ffaa00'
                  : '#44ff44',
              transition: 'all 0.3s ease',
              boxShadow: 'inset 0 2px 6px rgba(255, 255, 255, 0.3)',
            }}
          />
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              fontSize: '14px',
              fontWeight: 'bold',
              color: '#fff',
              textShadow: '1px 1px 2px rgba(0, 0, 0, 0.9)',
            }}
          >
            {depletionPercentage.toFixed(1)}%
          </div>
        </div>
      </div>

      {depletionPercentage < 25 && (
        <div
          className="warning"
          style={{
            marginTop: '12px',
            padding: '8px',
            backgroundColor: 'rgba(255, 68, 68, 0.2)',
            border: '1px solid #ff4444',
            borderRadius: '4px',
            fontSize: '13px',
            color: '#ff8888',
            textAlign: 'center',
          }}
        >
          ⚠ Mine nearly depleted!
        </div>
      )}

      {goldRemaining <= 0 && (
        <div
          className="depleted"
          style={{
            marginTop: '12px',
            padding: '8px',
            backgroundColor: 'rgba(100, 100, 100, 0.3)',
            border: '1px solid #666',
            borderRadius: '4px',
            fontSize: '13px',
            color: '#999',
            textAlign: 'center',
          }}
        >
          Mine Depleted
        </div>
      )}
    </div>
  );
};

export default GoldMine;
