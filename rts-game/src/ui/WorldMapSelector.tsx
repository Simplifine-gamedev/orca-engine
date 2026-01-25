/**
 * WorldMapSelector Component
 * Allows players to select a map for their RTS game
 */

import React, { useState, useMemo } from 'react';
import { MapPreset, MapSize } from '../types/MapTypes';
import { MAP_PRESETS, getMapsBySize } from '../config/mapPresets';

interface WorldMapSelectorProps {
  onMapSelect: (map: MapPreset) => void;
  selectedMapId?: string;
  maxPlayers?: number;
}

export const WorldMapSelector: React.FC<WorldMapSelectorProps> = ({
  onMapSelect,
  selectedMapId,
  maxPlayers,
}) => {
  const [selectedSize, setSelectedSize] = useState<MapSize | 'all'>('all');
  const [hoveredMapId, setHoveredMapId] = useState<string | null>(null);

  // Filter maps based on size and player count
  const filteredMaps = useMemo(() => {
    let maps = selectedSize === 'all' ? MAP_PRESETS : getMapsBySize(selectedSize);
    
    if (maxPlayers) {
      maps = maps.filter((map) => map.maxPlayers >= maxPlayers);
    }
    
    return maps;
  }, [selectedSize, maxPlayers]);

  const MapCard = ({ map }: { map: MapPreset }) => {
    const isSelected = selectedMapId === map.id;
    const isHovered = hoveredMapId === map.id;

    return (
      <div
        className={`map-card ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}`}
        onClick={() => onMapSelect(map)}
        onMouseEnter={() => setHoveredMapId(map.id)}
        onMouseLeave={() => setHoveredMapId(null)}
        style={{
          border: isSelected ? `3px solid ${map.previewColor}` : '1px solid #444',
          cursor: 'pointer',
          borderRadius: '8px',
          padding: '12px',
          backgroundColor: isHovered ? '#2a2a2a' : '#1a1a1a',
          transition: 'all 0.2s ease',
          transform: isHovered ? 'scale(1.05)' : 'scale(1)',
        }}
      >
        {/* Map Preview */}
        <div
          className="map-preview"
          style={{
            width: '100%',
            height: '150px',
            backgroundColor: map.previewColor,
            borderRadius: '4px',
            marginBottom: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Terrain visualization */}
          <div
            style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: `${map.terrain.water}%`,
              backgroundColor: 'rgba(59, 130, 246, 0.6)',
            }}
          />
          <div
            style={{
              position: 'absolute',
              fontSize: '48px',
              opacity: 0.3,
              fontWeight: 'bold',
              color: 'white',
            }}
          >
            {map.size.toUpperCase()}
          </div>
        </div>

        {/* Map Info */}
        <div style={{ color: 'white' }}>
          <h3 style={{ margin: '0 0 8px 0', fontSize: '16px', fontWeight: 'bold' }}>
            {map.name}
          </h3>
          <p style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#aaa' }}>
            {map.description}
          </p>
          
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', fontSize: '11px' }}>
            <span
              style={{
                padding: '2px 8px',
                backgroundColor: '#333',
                borderRadius: '4px',
                textTransform: 'uppercase',
              }}
            >
              {map.size}
            </span>
            <span
              style={{
                padding: '2px 8px',
                backgroundColor: '#333',
                borderRadius: '4px',
              }}
            >
              {map.width}x{map.height}
            </span>
            <span
              style={{
                padding: '2px 8px',
                backgroundColor: '#333',
                borderRadius: '4px',
              }}
            >
              {map.maxPlayers} Players
            </span>
            <span
              style={{
                padding: '2px 8px',
                backgroundColor: '#333',
                borderRadius: '4px',
                textTransform: 'capitalize',
              }}
            >
              {map.layout}
            </span>
          </div>

          {/* Terrain Stats */}
          <div style={{ marginTop: '10px', fontSize: '11px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ color: '#60A5FA' }}>Water: {map.terrain.water}%</span>
              <span style={{ color: '#22C55E' }}>Land: {map.terrain.land}%</span>
              <span style={{ color: '#9CA3AF' }}>Mountains: {map.terrain.mountains}%</span>
            </div>
          </div>

          {/* Resources Info */}
          {map.resources.high && (
            <div
              style={{
                marginTop: '8px',
                padding: '4px 8px',
                backgroundColor: '#166534',
                borderRadius: '4px',
                fontSize: '10px',
                textAlign: 'center',
                fontWeight: 'bold',
              }}
            >
              HIGH RESOURCES
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="world-map-selector" style={{ padding: '20px', color: 'white' }}>
      <h2 style={{ marginBottom: '20px', fontSize: '24px', fontWeight: 'bold' }}>
        Select Map
      </h2>

      {/* Size Filter */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ marginBottom: '10px', fontSize: '14px', fontWeight: 'bold' }}>
          Filter by Size:
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {['all', 'small', 'medium', 'large', 'huge'].map((size) => (
            <button
              key={size}
              onClick={() => setSelectedSize(size as MapSize | 'all')}
              style={{
                padding: '8px 16px',
                backgroundColor: selectedSize === size ? '#3B82F6' : '#333',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                textTransform: 'capitalize',
                fontWeight: selectedSize === size ? 'bold' : 'normal',
                transition: 'all 0.2s ease',
              }}
            >
              {size}
            </button>
          ))}
        </div>
      </div>

      {/* Map Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '20px',
          marginTop: '20px',
        }}
      >
        {filteredMaps.map((map) => (
          <MapCard key={map.id} map={map} />
        ))}
      </div>

      {/* No maps message */}
      {filteredMaps.length === 0 && (
        <div
          style={{
            textAlign: 'center',
            padding: '40px',
            color: '#aaa',
            fontSize: '16px',
          }}
        >
          No maps available for the selected filters.
        </div>
      )}

      {/* Selected Map Details */}
      {selectedMapId && (
        <div
          style={{
            marginTop: '30px',
            padding: '20px',
            backgroundColor: '#1a1a1a',
            borderRadius: '8px',
            border: '1px solid #444',
          }}
        >
          <h3 style={{ marginBottom: '10px', fontSize: '18px' }}>Selected Map Details</h3>
          {(() => {
            const selected = MAP_PRESETS.find((m) => m.id === selectedMapId);
            if (!selected) return null;
            
            return (
              <div style={{ fontSize: '14px' }}>
                <p>
                  <strong>Name:</strong> {selected.name}
                </p>
                <p>
                  <strong>Size:</strong> {selected.width}x{selected.height} ({selected.size})
                </p>
                <p>
                  <strong>Layout:</strong> {selected.layout}
                </p>
                <p>
                  <strong>Max Players:</strong> {selected.maxPlayers}
                </p>
                <p>
                  <strong>Resources:</strong> {selected.resources.high ? 'High' : 'Normal'} (
                  {selected.resources.distribution})
                </p>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default WorldMapSelector;
