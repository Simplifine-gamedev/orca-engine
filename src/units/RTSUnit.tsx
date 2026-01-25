// RTSUnit component with path visualization

import React from 'react';
import { Unit, Vector2 } from '../types';

interface RTSUnitProps {
  unit: Unit;
  showPath: boolean;
}

export const RTSUnit: React.FC<RTSUnitProps> = ({ unit, showPath }) => {
  return (
    <g>
      {/* Path visualization */}
      {showPath && unit.path.length > 1 && (
        <g>
          <polyline
            points={unit.path.map(p => `${p.x},${p.y}`).join(' ')}
            stroke={unit.selected ? '#4ade80' : '#94a3b8'}
            strokeWidth="2"
            strokeDasharray="5,5"
            fill="none"
            opacity="0.6"
          />
          {/* Target marker */}
          {unit.targetPosition && (
            <circle
              cx={unit.targetPosition.x}
              cy={unit.targetPosition.y}
              r="4"
              fill={unit.selected ? '#4ade80' : '#94a3b8'}
              opacity="0.6"
            />
          )}
        </g>
      )}

      {/* Unit body */}
      <g transform={`translate(${unit.position.x}, ${unit.position.y}) rotate(${unit.facingAngle * 180 / Math.PI})`}>
        {/* Main body */}
        <circle
          r="20"
          fill={unit.selected ? '#3b82f6' : '#64748b'}
          stroke={unit.selected ? '#93c5fd' : '#cbd5e1'}
          strokeWidth="2"
        />
        
        {/* Direction indicator */}
        <polygon
          points="15,0 -5,8 -5,-8"
          fill={unit.selected ? '#93c5fd' : '#cbd5e1'}
        />
        
        {/* Selection ring */}
        {unit.selected && (
          <circle
            r="25"
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
            opacity="0.5"
          />
        )}
      </g>
    </g>
  );
};

interface FormationPreviewProps {
  dragStart: Vector2;
  dragEnd: Vector2;
}

export const FormationPreview: React.FC<FormationPreviewProps> = ({ dragStart, dragEnd }) => {
  const dx = dragEnd.x - dragStart.x;
  const dy = dragEnd.y - dragStart.y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length < 5) return null;

  return (
    <g>
      {/* Direction line */}
      <line
        x1={dragStart.x}
        y1={dragStart.y}
        x2={dragEnd.x}
        y2={dragEnd.y}
        stroke="#fbbf24"
        strokeWidth="3"
        opacity="0.8"
      />
      
      {/* Arrowhead */}
      <polygon
        points={`${dragEnd.x},${dragEnd.y} ${dragEnd.x - 10},${dragEnd.y - 5} ${dragEnd.x - 10},${dragEnd.y + 5}`}
        fill="#fbbf24"
        opacity="0.8"
        transform={`rotate(${Math.atan2(dy, dx) * 180 / Math.PI}, ${dragEnd.x}, ${dragEnd.y})`}
      />
      
      {/* Center circle */}
      <circle
        cx={dragStart.x}
        cy={dragStart.y}
        r="8"
        fill="#fbbf24"
        opacity="0.6"
      />
    </g>
  );
};

interface GroupPathProps {
  path: Vector2[];
}

export const GroupPath: React.FC<GroupPathProps> = ({ path }) => {
  if (path.length < 2) return null;

  return (
    <g>
      <polyline
        points={path.map(p => `${p.x},${p.y}`).join(' ')}
        stroke="#f59e0b"
        strokeWidth="4"
        strokeDasharray="10,10"
        fill="none"
        opacity="0.7"
      />
      <circle
        cx={path[path.length - 1].x}
        cy={path[path.length - 1].y}
        r="8"
        fill="#f59e0b"
        opacity="0.7"
      />
    </g>
  );
};
