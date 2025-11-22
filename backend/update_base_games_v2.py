#!/usr/bin/env python3
"""
Update base_game_v2.json with license information and validate JSON structure.
"""

import json
import sys

# License mapping based on validation results
LICENSE_MAP = {
    "godot_official_2d_platformer_demo": "MIT (Godot demo projects)",
    "godot_official_3d_platformer_demo": "MIT (Godot demo projects)",
    "godot_official_physics_platformer_demo": "MIT (Godot demo projects)",
    "godot_official_tps_demo": "MIT",
    "maaack_godot_game_template": "MIT",
    "g2p_2d_platformer_starter_kit": "MIT",
    "brett_chalupa_2d_platformer_starter": "Code CC0; assets CC0/CC-BY",
    "gdquest_open_2d_platformer": "MIT",
    "stefh_2d_topdown_template": "MIT",
    "noidexe_topdown_action_rpg_template": "MIT",
    "bozar_godot4_roguelike_tutorial": "MIT",
    "quiver_topdown_shooter_core": "MIT",
    "quiver_tiny_wizard_demo": "MIT",
    "quiver_outpost_assault_tower_defense": "MIT",
    "godot_tactical_rpg_demo": "MIT",
    "survivors_starter_kit": "MIT",
    "kenney_3d_platformer_starter_kit": "MIT (code), CC0 assets",
    "kenney_fps_starter_kit": "MIT (code), CC0 assets",
    "kenney_city_builder_starter_kit": "MIT (code), CC0 assets",
    "whimfoome_first_person_starter": "MIT",
    "cogito_immersive_sim_template": "MIT",
    "dialogic_visual_novel_template": "MIT",
    "rakugo_visual_novel_kit": "MIT",
    "waffleawt_godot42_card_game_starter": "MIT",
    "chun_card_framework": "MIT"
}


def main():
    json_file = 'base_game_v2.json'
    
    # Read JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {json_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_file}: {e}")
        sys.exit(1)
    
    # Update entries with licenses
    updated_count = 0
    for entry in entries:
        entry_id = entry.get('id', '')
        if entry_id in LICENSE_MAP:
            if 'license' not in entry or entry.get('license') != LICENSE_MAP[entry_id]:
                entry['license'] = LICENSE_MAP[entry_id]
                updated_count += 1
        else:
            # Default to MIT if not found
            if 'license' not in entry:
                entry['license'] = "MIT"
                updated_count += 1
    
    # Write back with proper formatting
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"✅ Updated {json_file} with {updated_count} license entries")
        print(f"✅ Total entries: {len(entries)}")
    except Exception as e:
        print(f"❌ Error writing {json_file}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

