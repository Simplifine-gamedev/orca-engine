import { useEffect } from 'react';
import { useGameStore } from '../store/gameStore';

export const GameCanvas: React.FC = () => {
  const { units, addUnit, attackUnit } = useGameStore();

  // Initialize demo units
  useEffect(() => {
    if (units.length === 0) {
      addUnit({
        id: 'warrior-1',
        name: 'Warrior',
        health: 100,
        maxHealth: 100,
        attack: 25,
        x: 150,
        y: 200,
        team: 'ally',
      });

      addUnit({
        id: 'enemy-1',
        name: 'Goblin',
        health: 80,
        maxHealth: 80,
        attack: 15,
        x: 450,
        y: 200,
        team: 'enemy',
      });

      addUnit({
        id: 'mage-1',
        name: 'Mage',
        health: 60,
        maxHealth: 60,
        attack: 35,
        x: 150,
        y: 350,
        team: 'ally',
      });
    }
  }, [units.length, addUnit]);

  const handleUnitClick = (unitId: string) => {
    const unit = units.find((u) => u.id === unitId);
    if (!unit) return;

    // Demo: Attack the first enemy unit
    const enemy = units.find((u) => u.team !== unit.team);
    if (enemy) {
      attackUnit(unitId, enemy.id);
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '500px',
        backgroundColor: '#2a2a2a',
        border: '2px solid #444',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    >
      {units.map((unit) => (
        <div
          key={unit.id}
          onClick={() => handleUnitClick(unit.id)}
          style={{
            position: 'absolute',
            left: `${unit.x}px`,
            top: `${unit.y}px`,
            width: '60px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'transform 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'scale(1.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'scale(1)';
          }}
        >
          {/* Unit sprite */}
          <div
            style={{
              width: '50px',
              height: '50px',
              backgroundColor: unit.team === 'ally' ? '#4ADE80' : '#EF4444',
              borderRadius: '50%',
              border: '3px solid #fff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
            }}
          >
            {unit.team === 'ally' ? '⚔️' : '👹'}
          </div>

          {/* Health bar */}
          <div
            style={{
              marginTop: '5px',
              width: '100%',
              height: '8px',
              backgroundColor: '#333',
              borderRadius: '4px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${(unit.health / unit.maxHealth) * 100}%`,
                height: '100%',
                backgroundColor: unit.health > 30 ? '#4ADE80' : '#EF4444',
                transition: 'width 0.3s',
              }}
            />
          </div>

          {/* Unit name */}
          <div
            style={{
              marginTop: '2px',
              fontSize: '10px',
              color: '#fff',
              fontWeight: 'bold',
            }}
          >
            {unit.name}
          </div>

          {/* Health text */}
          <div
            style={{
              fontSize: '9px',
              color: '#aaa',
            }}
          >
            {unit.health}/{unit.maxHealth}
          </div>
        </div>
      ))}

      {/* Instructions */}
      <div
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          color: '#aaa',
          fontSize: '14px',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          padding: '10px',
          borderRadius: '4px',
        }}
      >
        Click on units to attack enemies
      </div>
    </div>
  );
};
