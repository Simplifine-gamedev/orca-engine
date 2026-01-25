import React from 'react';
import { useGameStore } from '../store/gameStore';
import { ResourceData } from '../types/resource';

export const SelectionPanel: React.FC = () => {
  const { selectedEntity, deselectEntity, assignWorker, unassignWorker } = useGameStore();
  
  if (!selectedEntity || selectedEntity.type !== 'resource') {
    return null;
  }
  
  const resource = selectedEntity.data as ResourceData;
  
  const getResourceIcon = (type: string) => {
    switch (type) {
      case 'gold_mine':
        return '⛏️';
      case 'tree':
        return '🌲';
      case 'stone_quarry':
        return '🪨';
      default:
        return '📦';
    }
  };
  
  const getResourceTypeName = (type: string) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };
  
  const fillPercentage = ((resource.amountRemaining / resource.maxAmount) * 100).toFixed(1);
  const totalGatherRate = resource.workersAssigned * resource.gatherRate;
  const canAssignMore = resource.workersAssigned < resource.maxWorkers;
  const canUnassign = resource.workersAssigned > 0;
  
  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: 'rgba(30, 30, 30, 0.95)',
        border: '2px solid #444',
        borderRadius: '12px',
        padding: '20px',
        minWidth: '400px',
        maxWidth: '500px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
        color: 'white',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        zIndex: 1000
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '32px' }}>{getResourceIcon(resource.type)}</span>
          <div>
            <h3 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>
              {getResourceTypeName(resource.type)}
            </h3>
            <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#999' }}>
              ID: {resource.id}
            </p>
          </div>
        </div>
        <button
          onClick={deselectEntity}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#999',
            fontSize: '24px',
            cursor: 'pointer',
            padding: '4px 8px',
            transition: 'color 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'white'}
          onMouseLeave={(e) => e.currentTarget.style.color = '#999'}
        >
          ×
        </button>
      </div>
      
      {/* Resource Amount */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ fontSize: '14px', color: '#ccc' }}>Amount Remaining</span>
          <span style={{ fontSize: '14px', fontWeight: 'bold' }}>
            {resource.amountRemaining} / {resource.maxAmount} ({fillPercentage}%)
          </span>
        </div>
        <div
          style={{
            width: '100%',
            height: '12px',
            backgroundColor: '#333',
            borderRadius: '6px',
            overflow: 'hidden',
            border: '1px solid #555'
          }}
        >
          <div
            style={{
              width: `${fillPercentage}%`,
              height: '100%',
              backgroundColor: resource.type === 'gold_mine' ? '#FFD700' : 
                             resource.type === 'tree' ? '#228B22' : '#808080',
              transition: 'width 0.3s ease',
              boxShadow: 'inset 0 2px 4px rgba(255, 255, 255, 0.2)'
            }}
          />
        </div>
      </div>
      
      {/* Workers Section */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '14px', color: '#ccc' }}>Workers Assigned</span>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button
              onClick={() => unassignWorker(resource.id)}
              disabled={!canUnassign}
              style={{
                backgroundColor: canUnassign ? '#c44' : '#444',
                border: 'none',
                color: 'white',
                width: '28px',
                height: '28px',
                borderRadius: '4px',
                cursor: canUnassign ? 'pointer' : 'not-allowed',
                fontSize: '18px',
                fontWeight: 'bold',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => canUnassign && (e.currentTarget.style.backgroundColor = '#e44')}
              onMouseLeave={(e) => canUnassign && (e.currentTarget.style.backgroundColor = '#c44')}
            >
              −
            </button>
            <span style={{ fontSize: '16px', fontWeight: 'bold', minWidth: '60px', textAlign: 'center' }}>
              {resource.workersAssigned} / {resource.maxWorkers}
            </span>
            <button
              onClick={() => assignWorker(resource.id)}
              disabled={!canAssignMore}
              style={{
                backgroundColor: canAssignMore ? '#4c4' : '#444',
                border: 'none',
                color: 'white',
                width: '28px',
                height: '28px',
                borderRadius: '4px',
                cursor: canAssignMore ? 'pointer' : 'not-allowed',
                fontSize: '18px',
                fontWeight: 'bold',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => canAssignMore && (e.currentTarget.style.backgroundColor = '#5e5')}
              onMouseLeave={(e) => canAssignMore && (e.currentTarget.style.backgroundColor = '#4c4')}
            >
              +
            </button>
          </div>
        </div>
        
        {/* Worker icons */}
        <div style={{ display: 'flex', gap: '4px', justifyContent: 'center' }}>
          {Array.from({ length: resource.maxWorkers }).map((_, i) => (
            <div
              key={i}
              style={{
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                backgroundColor: i < resource.workersAssigned ? '#4c4' : '#333',
                border: '2px solid ' + (i < resource.workersAssigned ? '#5e5' : '#555'),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '12px',
                transition: 'all 0.3s ease'
              }}
            >
              {i < resource.workersAssigned ? '👷' : ''}
            </div>
          ))}
        </div>
      </div>
      
      {/* Gather Rate */}
      <div
        style={{
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '8px',
          padding: '12px',
          border: '1px solid #444'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span style={{ fontSize: '14px', color: '#ccc' }}>Gather Rate (per worker)</span>
          <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#FFD700' }}>
            {resource.gatherRate}/s
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '14px', color: '#ccc' }}>Total Production</span>
          <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#4c4' }}>
            {totalGatherRate.toFixed(1)}/s
          </span>
        </div>
      </div>
      
      {/* Location Info */}
      <div style={{ marginTop: '12px', fontSize: '12px', color: '#666', textAlign: 'center' }}>
        Position: ({resource.position.x}, {resource.position.y})
      </div>
    </div>
  );
};
