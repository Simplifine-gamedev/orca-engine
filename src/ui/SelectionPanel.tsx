import React from 'react';
import { useGameStore } from '../store/gameStore';

export const SelectionPanel: React.FC = () => {
  const { resources, selectedResourceId, addWorkerToResource, removeWorkerFromResource } = useGameStore();
  
  const selectedResource = resources.find((r) => r.id === selectedResourceId);

  if (!selectedResource) {
    return (
      <div style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '320px',
        backgroundColor: 'rgba(30, 30, 40, 0.95)',
        border: '2px solid #555',
        borderRadius: '12px',
        padding: '20px',
        color: 'white',
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        fontFamily: 'Arial, sans-serif'
      }}>
        <h3 style={{ margin: '0 0 10px 0', fontSize: '18px', color: '#AAA' }}>
          Selection Info
        </h3>
        <p style={{ margin: '0', fontSize: '14px', color: '#888' }}>
          Click on a resource to view details
        </p>
      </div>
    );
  }

  const getResourceTypeDisplay = () => {
    switch (selectedResource.type) {
      case 'goldmine':
        return 'Gold Mine ⛏️';
      case 'tree':
        return 'Tree 🌲';
      default:
        return 'Unknown Resource';
    }
  };

  const getResourceColor = () => {
    switch (selectedResource.type) {
      case 'goldmine':
        return '#FFD700';
      case 'tree':
        return '#228B22';
      default:
        return '#808080';
    }
  };

  const percentRemaining = Math.round((selectedResource.amountRemaining / selectedResource.maxAmount) * 100);

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      width: '320px',
      backgroundColor: 'rgba(30, 30, 40, 0.95)',
      border: `2px solid ${getResourceColor()}`,
      borderRadius: '12px',
      padding: '20px',
      color: 'white',
      boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h3 style={{ 
        margin: '0 0 15px 0', 
        fontSize: '20px', 
        color: getResourceColor(),
        borderBottom: `2px solid ${getResourceColor()}`,
        paddingBottom: '10px'
      }}>
        {getResourceTypeDisplay()}
      </h3>
      
      <div style={{ marginBottom: '15px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginBottom: '8px',
          fontSize: '14px'
        }}>
          <span style={{ color: '#CCC' }}>Resource ID:</span>
          <span style={{ color: '#FFF', fontWeight: 'bold' }}>{selectedResource.id}</span>
        </div>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginBottom: '8px',
          fontSize: '14px'
        }}>
          <span style={{ color: '#CCC' }}>Amount Remaining:</span>
          <span style={{ color: '#FFF', fontWeight: 'bold' }}>
            {selectedResource.amountRemaining} / {selectedResource.maxAmount}
          </span>
        </div>
        
        <div style={{ marginBottom: '12px' }}>
          <div style={{
            width: '100%',
            height: '8px',
            backgroundColor: '#333',
            borderRadius: '4px',
            overflow: 'hidden',
            border: '1px solid #555'
          }}>
            <div style={{
              width: `${percentRemaining}%`,
              height: '100%',
              backgroundColor: getResourceColor(),
              transition: 'width 0.3s ease'
            }} />
          </div>
          <div style={{ 
            fontSize: '12px', 
            color: '#AAA', 
            textAlign: 'right',
            marginTop: '4px'
          }}>
            {percentRemaining}% remaining
          </div>
        </div>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginBottom: '8px',
          fontSize: '14px'
        }}>
          <span style={{ color: '#CCC' }}>Workers Assigned:</span>
          <span style={{ color: '#FFF', fontWeight: 'bold' }}>
            {selectedResource.workersAssigned}
          </span>
        </div>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginBottom: '8px',
          fontSize: '14px'
        }}>
          <span style={{ color: '#CCC' }}>Gather Rate:</span>
          <span style={{ color: '#FFF', fontWeight: 'bold' }}>
            {selectedResource.gatherRate * selectedResource.workersAssigned}/sec
          </span>
        </div>
      </div>

      <div style={{ 
        display: 'flex', 
        gap: '10px',
        marginTop: '15px',
        paddingTop: '15px',
        borderTop: '1px solid #555'
      }}>
        <button
          onClick={() => addWorkerToResource(selectedResource.id)}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: '#28A745',
            border: 'none',
            borderRadius: '6px',
            color: 'white',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#218838';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = '#28A745';
          }}
        >
          + Add Worker
        </button>
        
        <button
          onClick={() => removeWorkerFromResource(selectedResource.id)}
          disabled={selectedResource.workersAssigned === 0}
          style={{
            flex: 1,
            padding: '10px',
            backgroundColor: selectedResource.workersAssigned === 0 ? '#555' : '#DC3545',
            border: 'none',
            borderRadius: '6px',
            color: 'white',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: selectedResource.workersAssigned === 0 ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s',
            opacity: selectedResource.workersAssigned === 0 ? 0.5 : 1
          }}
          onMouseEnter={(e) => {
            if (selectedResource.workersAssigned > 0) {
              e.currentTarget.style.backgroundColor = '#C82333';
            }
          }}
          onMouseLeave={(e) => {
            if (selectedResource.workersAssigned > 0) {
              e.currentTarget.style.backgroundColor = '#DC3545';
            }
          }}
        >
          - Remove Worker
        </button>
      </div>
    </div>
  );
};
