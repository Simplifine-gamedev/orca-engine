# Generated Factions Data

This directory contains auto-generated faction configuration data with complete thumbnail URLs for all buildings and units.

## Files

### factions_summary.json

Complete faction data including:
- Unit stats and thumbnails
- Building stats and thumbnails
- Cost and build time information
- Resource production data

## Thumbnail Generation

Thumbnails are stored in `/rts-game/public/assets/`:

```
assets/
├── units/
│   ├── human/
│   │   ├── footman_preview.png
│   │   ├── archer_preview.png
│   │   └── knight_preview.png
│   ├── dwarf/
│   │   ├── warrior_preview.png
│   │   ├── rifleman_preview.png
│   │   └── hammerer_preview.png
│   └── undead/
│       ├── worker_thumbnail.png
│       ├── soldier_light_thumbnail.png
│       ├── soldier_medium_thumbnail.png
│       └── soldier_heavy_thumbnail.png
└── buildings/
    ├── human/
    │   └── barracks_thumbnail.png
    ├── dwarf/
    │   └── barracks_thumbnail.png
    └── undead/
        ├── city_center_thumbnail.png
        ├── barracks_thumbnail.png
        ├── farm_thumbnail.png
        ├── bank_thumbnail.png
        ├── mill_thumbnail.png
        ├── warehouse_thumbnail.png
        └── tower_thumbnail.png
```

## Regenerating Thumbnails

To regenerate thumbnails from GLB models:

```bash
# From 3D models (requires trimesh, pyrender, pillow)
python3 scripts/generate_thumbnails.py --faction-config generated_factions/factions_summary.json

# Generate placeholders (for development)
python3 scripts/generate_placeholder_thumbnails.py --output-dir ./public/assets
```

## Faction Overview

### Human
- **Units**: Footman, Archer, Knight
- **Buildings**: Barracks
- **Strength**: Balanced and versatile

### Dwarf
- **Units**: Warrior, Rifleman, Hammerer
- **Buildings**: Barracks
- **Strength**: High defense and ranged damage

### Undead
- **Units**: Worker, Skeleton Warrior, Zombie Soldier, Death Knight
- **Buildings**: Necropolis, Crypt, Graveyard, Haunted Treasury, Bone Mill, Tomb Storage, Spirit Tower
- **Strength**: Resource generation and defensive structures
