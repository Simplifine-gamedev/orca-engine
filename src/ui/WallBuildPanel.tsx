import React, { useState, useEffect } from 'react';
import { WallSystem, WallSegment } from '../buildings/WallSystem';

export interface WallBuildPanelProps {
  initialResources?: number;
  onClose?: () => void;
}

export const WallBuildPanel: React.FC<WallBuildPanelProps> = ({
  initialResources = 1000,
  onClose,
}) => {
  const [resources, setResources] = useState(initialResources);
  const [placedWalls, setPlacedWalls] = useState<WallSegment[]>([]);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [buildMode, setBuildMode] = useState<'single' | 'continuous'>('single');
  const [showStats, setShowStats] = useState(true);

  const handleWallPlaced = (segment: WallSegment) => {
    setResources(prev => prev - segment.cost);
    setPlacedWalls(prev => [...prev, segment]);
    
    // Show success feedback
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 200, 0, 0.95);
      color: white;
      padding: 12px 20px;
      border-radius: 6px;
      font-weight: bold;
      z-index: 10000;
      animation: slideDown 0.3s ease;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    `;
    notification.textContent = `Wall placed! Cost: ${segment.cost}`;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.style.animation = 'slideUp 0.3s ease';
      setTimeout(() => document.body.removeChild(notification), 300);
    }, 2000);
  };

  const handleCancelled = () => {
    console.log('Wall placement cancelled');
  };

  const handleUndo = () => {
    if (placedWalls.length === 0) return;
    
    const lastWall = placedWalls[placedWalls.length - 1];
    setResources(prev => prev + lastWall.cost);
    setPlacedWalls(prev => prev.slice(0, -1));
  };

  const handleClearAll = () => {
    if (!window.confirm('Are you sure you want to remove all walls?')) return;
    
    const totalRefund = placedWalls.reduce((sum, wall) => sum + wall.cost, 0);
    setResources(prev => prev + totalRefund);
    setPlacedWalls([]);
  };

  const totalSpent = placedWalls.reduce((sum, wall) => sum + wall.cost, 0);
  const totalWallLength = placedWalls.length;

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      gap: '20px',
      padding: '20px',
      background: '#f5f5f5',
      minHeight: '100vh',
    }}>
      {/* Header */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ margin: 0, fontSize: '24px', color: '#333' }}>Wall Builder</h1>
          {onClose && (
            <button
              onClick={onClose}
              style={{
                padding: '8px 16px',
                background: '#666',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
              }}
            >
              Close
            </button>
          )}
        </div>
        
        <div style={{ 
          marginTop: '15px', 
          display: 'flex', 
          gap: '20px',
          flexWrap: 'wrap',
        }}>
          <div style={{ 
            background: '#e8f5e9', 
            padding: '12px 16px', 
            borderRadius: '6px',
            flex: '1',
            minWidth: '150px',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>Resources</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2e7d32' }}>
              {resources}
            </div>
          </div>
          
          <div style={{ 
            background: '#e3f2fd', 
            padding: '12px 16px', 
            borderRadius: '6px',
            flex: '1',
            minWidth: '150px',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>Total Spent</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1565c0' }}>
              {totalSpent}
            </div>
          </div>
          
          <div style={{ 
            background: '#fff3e0', 
            padding: '12px 16px', 
            borderRadius: '6px',
            flex: '1',
            minWidth: '150px',
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>Walls Placed</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#e65100' }}>
              {totalWallLength}
            </div>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      }}>
        <h2 style={{ margin: '0 0 15px 0', fontSize: '18px', color: '#333' }}>Controls</h2>
        
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={handleUndo}
            disabled={placedWalls.length === 0}
            style={{
              padding: '10px 20px',
              background: placedWalls.length > 0 ? '#ff9800' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: placedWalls.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            Undo Last Wall
          </button>
          
          <button
            onClick={handleClearAll}
            disabled={placedWalls.length === 0}
            style={{
              padding: '10px 20px',
              background: placedWalls.length > 0 ? '#f44336' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: placedWalls.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            Clear All
          </button>
          
          <button
            onClick={() => setShowStats(!showStats)}
            style={{
              padding: '10px 20px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
            }}
          >
            {showStats ? 'Hide' : 'Show'} Stats
          </button>
        </div>

        {/* Quick Tips */}
        <div style={{
          marginTop: '20px',
          padding: '15px',
          background: '#f0f7ff',
          borderLeft: '4px solid #2196F3',
          borderRadius: '4px',
        }}>
          <h3 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#1976d2' }}>
            Quick Tips
          </h3>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: '#555' }}>
            <li>Left-click to place wall points</li>
            <li><strong>Right-click to cancel</strong> current wall</li>
            <li>Green areas show valid placement zones</li>
            <li>Cost preview updates in real-time</li>
            <li>Walls cannot be placed on obstacles or existing structures</li>
          </ul>
        </div>
      </div>

      {/* Wall System Canvas */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      }}>
        <h2 style={{ margin: '0 0 15px 0', fontSize: '18px', color: '#333' }}>Build Area</h2>
        <WallSystem
          resources={resources}
          onWallPlaced={handleWallPlaced}
          onCancelled={handleCancelled}
          costPerUnit={10}
          gridSize={20}
        />
      </div>

      {/* Statistics Panel */}
      {showStats && (
        <div style={{
          background: 'white',
          padding: '20px',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        }}>
          <h2 style={{ margin: '0 0 15px 0', fontSize: '18px', color: '#333' }}>Wall Statistics</h2>
          
          {placedWalls.length === 0 ? (
            <p style={{ color: '#999', margin: 0 }}>No walls placed yet</p>
          ) : (
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #ddd' }}>
                    <th style={{ padding: '8px', textAlign: 'left', fontSize: '12px', color: '#666' }}>
                      #
                    </th>
                    <th style={{ padding: '8px', textAlign: 'left', fontSize: '12px', color: '#666' }}>
                      From
                    </th>
                    <th style={{ padding: '8px', textAlign: 'left', fontSize: '12px', color: '#666' }}>
                      To
                    </th>
                    <th style={{ padding: '8px', textAlign: 'right', fontSize: '12px', color: '#666' }}>
                      Cost
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {placedWalls.map((wall, index) => (
                    <tr key={wall.id} style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '8px', fontSize: '13px' }}>{index + 1}</td>
                      <td style={{ padding: '8px', fontSize: '13px' }}>
                        ({wall.startX}, {wall.startY})
                      </td>
                      <td style={{ padding: '8px', fontSize: '13px' }}>
                        ({wall.endX}, {wall.endY})
                      </td>
                      <td style={{ padding: '8px', fontSize: '13px', textAlign: 'right', fontWeight: 'bold' }}>
                        {wall.cost}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Keyboard Shortcuts Reference */}
      <div style={{
        background: 'white',
        padding: '15px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        fontSize: '12px',
        color: '#666',
      }}>
        <strong>Improved Controls:</strong> Right-click anywhere to cancel wall placement (ESC is no longer needed!)
      </div>
    </div>
  );
};

export default WallBuildPanel;
