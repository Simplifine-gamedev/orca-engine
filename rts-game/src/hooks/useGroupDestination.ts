import { useEffect, useState } from 'react';
import { useGameStore } from '../store/gameStore';

export const useGroupDestination = () => {
  const {
    units,
    pathVisibilityMode,
    groupDestinationMarkerEnabled,
    calculateGroupDestination
  } = useGameStore();
  
  const [groupDestination, setGroupDestination] = useState<{ x: number; y: number; z: number } | null>(null);
  const [selectedUnitCount, setSelectedUnitCount] = useState(0);
  
  useEffect(() => {
    const selectedUnits = units.filter(unit => unit.isSelected);
    setSelectedUnitCount(selectedUnits.length);
    
    // Only show group destination marker when:
    // 1. Mode is set to 'group-marker'
    // 2. Feature is enabled
    // 3. Multiple units are selected
    // 4. At least one unit has a destination
    if (
      pathVisibilityMode === 'group-marker' &&
      groupDestinationMarkerEnabled &&
      selectedUnits.length > 1
    ) {
      const destination = calculateGroupDestination();
      setGroupDestination(destination);
    } else {
      setGroupDestination(null);
    }
  }, [units, pathVisibilityMode, groupDestinationMarkerEnabled, calculateGroupDestination]);
  
  return {
    groupDestination,
    selectedUnitCount,
    shouldShowGroupMarker: groupDestination !== null
  };
};
