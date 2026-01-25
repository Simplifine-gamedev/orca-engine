import React, { useState } from 'react';
import { WallBuildPanel } from '../ui/WallBuildPanel';
import { WallSystem, WallSegment } from '../buildings/WallSystem';

/**
 * Demo component showing different ways to use the Wall Building System
 */
export const WallSystemDemo: React.FC = () => {
  const [demoMode, setDemoMode] = useState<'full' | 'minimal' | 'custom'>('full');
  const [customResources, setCustomResources] = useState(1000);

  return (
    <div style={{ 
      padding: '20px', 
      fontFamily: 'Arial, sans-serif',
      maxWidth: '1200px',
      margin: '0 auto',
    }}>
      <header style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '30px',
        borderRadius: '8px',
        marginBottom: '30px',
      }}>
        <h1 style={{ margin: '0 0 10px 0' }}>Wall Building System Demo</h1>
        <p style={{ margin: 0, opacity: 0.9 }}>
          Showcasing the improved UX for wall building in Orca RTS
        </p>
      </header>

      {/* Demo Mode Selector */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        marginBottom: '20px',
      }}>
        <h2 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>Select Demo Mode</h2>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={() => setDemoMode('full')}
            style={{
              padding: '12px 24px',
              background: demoMode === 'full' ? '#667eea' : '#f0f0f0',
              color: demoMode === 'full' ? 'white' : '#333',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: demoMode === 'full' ? 'bold' : 'normal',
              fontSize: '14px',
            }}
          >
            Full Panel
          </button>
          <button
            onClick={() => setDemoMode('minimal')}
            style={{
              padding: '12px 24px',
              background: demoMode === 'minimal' ? '#667eea' : '#f0f0f0',
              color: demoMode === 'minimal' ? 'white' : '#333',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: demoMode === 'minimal' ? 'bold' : 'normal',
              fontSize: '14px',
            }}
          >
            Minimal View
          </button>
          <button
            onClick={() => setDemoMode('custom')}
            style={{
              padding: '12px 24px',
              background: demoMode === 'custom' ? '#667eea' : '#f0f0f0',
              color: demoMode === 'custom' ? 'white' : '#333',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: demoMode === 'custom' ? 'bold' : 'normal',
              fontSize: '14px',
            }}
          >
            Custom Setup
          </button>
        </div>
      </div>

      {/* Key Features Highlight */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        marginBottom: '20px',
      }}>
        <h2 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>Key UX Improvements</h2>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '15px',
        }}>
          <div style={{ padding: '15px', background: '#e8f5e9', borderRadius: '6px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>🖱️</div>
            <strong>Right-click Cancel</strong>
            <p style={{ margin: '5px 0 0', fontSize: '13px', color: '#666' }}>
              No more confusing ESC key
            </p>
          </div>
          <div style={{ padding: '15px', background: '#e3f2fd', borderRadius: '6px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>💰</div>
            <strong>Cost Preview</strong>
            <p style={{ margin: '5px 0 0', fontSize: '13px', color: '#666' }}>
              See cost before confirming
            </p>
          </div>
          <div style={{ padding: '15px', background: '#fff3e0', borderRadius: '6px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>✅</div>
            <strong>Valid Areas</strong>
            <p style={{ margin: '5px 0 0', fontSize: '13px', color: '#666' }}>
              Highlighted buildable zones
            </p>
          </div>
          <div style={{ padding: '15px', background: '#f3e5f5', borderRadius: '6px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>📚</div>
            <strong>Tutorial Tooltip</strong>
            <p style={{ margin: '5px 0 0', fontSize: '13px', color: '#666' }}>
              First-time user guidance
            </p>
          </div>
        </div>
      </div>

      {/* Demo Content */}
      {demoMode === 'full' && (
        <WallBuildPanel 
          initialResources={1000}
          onClose={() => console.log('Panel closed')}
        />
      )}

      {demoMode === 'minimal' && (
        <div style={{
          background: 'white',
          padding: '20px',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        }}>
          <h2 style={{ margin: '0 0 15px 0' }}>Minimal Wall System</h2>
          <WallSystem
            resources={500}
            onWallPlaced={(segment) => console.log('Wall placed:', segment)}
            onCancelled={() => console.log('Cancelled')}
          />
        </div>
      )}

      {demoMode === 'custom' && (
        <div style={{
          background: 'white',
          padding: '20px',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        }}>
          <h2 style={{ margin: '0 0 15px 0' }}>Custom Configuration</h2>
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
              Starting Resources: {customResources}
            </label>
            <input
              type="range"
              min="100"
              max="5000"
              step="100"
              value={customResources}
              onChange={(e) => setCustomResources(Number(e.target.value))}
              style={{ width: '100%', maxWidth: '400px' }}
            />
          </div>
          <WallBuildPanel 
            initialResources={customResources}
          />
        </div>
      )}

      {/* Usage Instructions */}
      <div style={{
        background: '#f5f5f5',
        padding: '20px',
        borderRadius: '8px',
        marginTop: '20px',
        borderLeft: '4px solid #667eea',
      }}>
        <h3 style={{ margin: '0 0 10px 0', fontSize: '16px' }}>How to Use</h3>
        <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '14px', lineHeight: '1.8' }}>
          <li><strong>Click once</strong> to set the wall start point (blue square)</li>
          <li><strong>Move your mouse</strong> to see the wall preview (dashed line)</li>
          <li><strong>Click again</strong> to confirm and place the wall</li>
          <li><strong>Right-click</strong> to cancel at any time (no more ESC!)</li>
          <li><strong>Green areas</strong> show where walls can be built</li>
          <li><strong>Red preview</strong> means invalid placement or insufficient resources</li>
        </ol>
      </div>

      {/* Technical Notes */}
      <div style={{
        background: 'white',
        padding: '20px',
        borderRadius: '8px',
        marginTop: '20px',
        fontSize: '13px',
        color: '#666',
      }}>
        <h3 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#333' }}>Technical Notes</h3>
        <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>Canvas-based rendering for optimal performance</li>
          <li>Grid system with configurable size (default 20px)</li>
          <li>Real-time cost calculation based on distance</li>
          <li>localStorage-based tutorial state management</li>
          <li>Responsive design with flexbox layouts</li>
          <li>TypeScript for type safety</li>
        </ul>
      </div>
    </div>
  );
};

export default WallSystemDemo;
