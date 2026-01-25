import React from 'react';
import { render, screen } from '@testing-library/react';
import { ResourceBar } from '../ResourceBar';
import { useGameStore } from '../../store/gameStore';

// Mock the game store
jest.mock('../../store/gameStore');

describe('ResourceBar', () => {
  beforeEach(() => {
    // Reset mock before each test
    jest.clearAllMocks();
  });

  it('should display player faction population, not world population', () => {
    // Setup mock data
    const mockUseGameStore = useGameStore as jest.MockedFunction<typeof useGameStore>;
    
    mockUseGameStore.mockImplementation((selector: any) => {
      const state = {
        getPlayerPopulation: () => 25, // Player has 25 units
        getPlayerMaxPopulation: () => 100,
        getPlayerFaction: () => ({
          id: 'player1',
          name: 'Blue Team',
          color: '#0066CC',
          isPlayer: true
        }),
        getWorldPopulation: () => 150 // Total world has 150 units (should NOT be shown)
      };
      return selector(state);
    });

    render(<ResourceBar />);
    
    // Should show player population (25), not world population (150)
    expect(screen.getByText('25 / 100')).toBeInTheDocument();
    expect(screen.queryByText('150')).not.toBeInTheDocument();
  });

  it('should display faction information', () => {
    const mockUseGameStore = useGameStore as jest.MockedFunction<typeof useGameStore>;
    
    mockUseGameStore.mockImplementation((selector: any) => {
      const state = {
        getPlayerPopulation: () => 15,
        getPlayerMaxPopulation: () => 50,
        getPlayerFaction: () => ({
          id: 'player1',
          name: 'Red Army',
          color: '#CC0000',
          isPlayer: true
        })
      };
      return selector(state);
    });

    render(<ResourceBar />);
    
    expect(screen.getByText('Red Army')).toBeInTheDocument();
    expect(screen.getByText('15 / 50')).toBeInTheDocument();
  });

  it('should handle zero population correctly', () => {
    const mockUseGameStore = useGameStore as jest.MockedFunction<typeof useGameStore>;
    
    mockUseGameStore.mockImplementation((selector: any) => {
      const state = {
        getPlayerPopulation: () => 0,
        getPlayerMaxPopulation: () => 100,
        getPlayerFaction: () => null
      };
      return selector(state);
    });

    render(<ResourceBar />);
    
    expect(screen.getByText('0 / 100')).toBeInTheDocument();
  });
});
