import React from 'react';
import { Unit, UnitTeam } from '../types';

interface UnitInfoProps {
  units: Unit[];
  selectedUnitIds: string[];
}

export const UnitInfo: React.FC<UnitInfoProps> = ({ units, selectedUnitIds }) => {
  const selectedUnits = units.filter(u => selectedUnitIds.includes(u.id));
  
  if (selectedUnits.length === 0) {
    return (
      <div className="unit-info">
        <h3>No units selected</h3>
        <p>Click or drag to select units</p>
      </div>
    );
  }

  const friendlyUnits = selectedUnits.filter(u => u.team === UnitTeam.FRIENDLY);
  const enemyUnits = selectedUnits.filter(u => u.team === UnitTeam.ENEMY);
  
  return (
    <div className="unit-info">
      <h3>Selected Units ({selectedUnits.length})</h3>
      
      {friendlyUnits.length > 0 && (
        <div className="unit-group friendly">
          <h4 style={{ color: '#00FF00' }}>Friendly Units ({friendlyUnits.length})</h4>
          {friendlyUnits.map(unit => (
            <div key={unit.id} className="unit-details">
              <div><strong>{unit.name}</strong></div>
              <div>Health: {unit.health}/{unit.maxHealth}</div>
              <div>Position: ({Math.round(unit.position.x)}, {Math.round(unit.position.y)})</div>
            </div>
          ))}
          <div className="unit-commands">
            <button>Move</button>
            <button>Attack</button>
            <button>Stop</button>
          </div>
        </div>
      )}
      
      {enemyUnits.length > 0 && (
        <div className="unit-group enemy">
          <h4 style={{ color: '#FF0000' }}>Enemy Units ({enemyUnits.length})</h4>
          {enemyUnits.map(unit => (
            <div key={unit.id} className="unit-details">
              <div><strong>{unit.name}</strong></div>
              <div>Health: {unit.health}/{unit.maxHealth}</div>
              <div>Position: ({Math.round(unit.position.x)}, {Math.round(unit.position.y)})</div>
            </div>
          ))}
          <div className="info-notice" style={{ 
            color: '#FFA500', 
            padding: '8px', 
            marginTop: '8px',
            border: '1px solid #FFA500',
            borderRadius: '4px'
          }}>
            Enemy units cannot receive commands. Select for targeting info only.
          </div>
        </div>
      )}
    </div>
  );
};
