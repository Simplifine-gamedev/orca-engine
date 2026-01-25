# Generated Factions - Scout Units

This directory contains faction-specific scout unit configurations for Orca RTS.

## Scout Unit Overview

Scout units are designed for early game map exploration with the following characteristics:

- **Fast Movement**: Higher speed than combat units for quick map coverage
- **Large Vision Range**: Extended sight radius to reveal more of the map
- **Low Cost**: Affordable in early game to enable quick exploration
- **Low Combat Power**: Not designed for fighting, focused on reconnaissance

## Available Scout Units

### Human Scout
- **Cost**: 50 food, 25 gold
- **Speed**: 8.5
- **Vision**: 12
- **Special**: Keen Sight - Detects hidden units

### Elf Ranger
- **Cost**: 55 food, 30 gold, 10 wood
- **Speed**: 9.0 (fastest scout)
- **Vision**: 14 (best vision)
- **Special**: Forest Stealth - Invisible in forests
- **Bonus**: Forest movement speed increased by 20%

### Orc Warg Rider
- **Cost**: 60 food, 20 gold
- **Speed**: 8.0
- **Vision**: 10
- **Health**: 80 (most durable scout)
- **Special**: Intimidate and Savage Charge - Combat-oriented abilities
- **Bonus**: Night vision - No vision penalty at night

### Undead Shade
- **Cost**: 45 food, 35 gold
- **Speed**: 8.8
- **Vision**: 13
- **Special**: Phase Walk - Moves through obstacles
- **Special**: Shadowmeld - Temporary invisibility
- **Bonus**: Regenerates health on blight

## Usage

These JSON configurations can be loaded by the game engine to create faction-specific scout variants. Each faction's scout has unique strengths that align with their overall playstyle:

- **Humans**: Balanced and reliable
- **Elves**: Fast and stealthy
- **Orcs**: Durable and aggressive
- **Undead**: Evasive and supernatural

## Integration

To add these scouts to your game:

1. Import the JSON files in your faction loader
2. Register units with the game store
3. Make available from appropriate buildings (town center, stable, etc.)
4. Implement special abilities in your game logic

## Balance Notes

Scout units should be:
- Available very early (first building)
- Cost less than combat units
- Die quickly in combat
- Excel at vision and speed

This encourages players to build scouts for exploration while still needing combat units for fighting.
