extends Node
class_name FactionsConfig

# Factions configuration for RTS game
# Defines all factions, their units, and properties

const FACTION_DATA = {
	"human": {
		"id": 0,
		"name": "Human Kingdom",
		"color": Color(0.2, 0.4, 0.8),
		"units": {
			"archer": {
				"name": "Archer",
				"type": "ranged_infantry",
				"health": 60,
				"damage": 12,
				"range": 15,
				"speed": 3.5,
				"attack_cooldown": 1.5,
				"cost_wood": 50,
				"cost_gold": 25,
				"train_time": 30,
				"trained_at": "archery_range",
				"model": "res://rts_game/units/archer.tscn",
				"icon": "res://rts_game/assets/icons/archer_icon.png"
			},
			"crossbowman": {
				"name": "Crossbowman",
				"type": "ranged_infantry",
				"health": 70,
				"damage": 18,
				"range": 18,
				"speed": 3.0,
				"attack_cooldown": 2.2,
				"armor_penetration": 5,
				"cost_wood": 60,
				"cost_gold": 40,
				"train_time": 45,
				"trained_at": "archery_range",
				"model": "res://rts_game/units/crossbowman.tscn",
				"icon": "res://rts_game/assets/icons/crossbowman_icon.png"
			}
		},
		"buildings": {
			"archery_range": {
				"name": "Archery Range",
				"type": "military",
				"health": 500,
				"cost_wood": 150,
				"cost_gold": 50,
				"build_time": 60,
				"trains": ["archer", "crossbowman"],
				"model": "res://rts_game/buildings/archery_range.tscn",
				"icon": "res://rts_game/assets/icons/archery_range_icon.png"
			}
		}
	},
	"orc": {
		"id": 1,
		"name": "Orc Horde",
		"color": Color(0.8, 0.2, 0.2),
		"units": {
			"archer": {
				"name": "Orc Archer",
				"type": "ranged_infantry",
				"health": 65,
				"damage": 13,
				"range": 14,
				"speed": 3.3,
				"attack_cooldown": 1.6,
				"cost_wood": 50,
				"cost_gold": 25,
				"train_time": 28,
				"trained_at": "archery_range",
				"model": "res://rts_game/units/archer.tscn",
				"icon": "res://rts_game/assets/icons/orc_archer_icon.png"
			},
			"crossbowman": {
				"name": "Orc Crossbowman",
				"type": "ranged_infantry",
				"health": 75,
				"damage": 20,
				"range": 17,
				"speed": 2.8,
				"attack_cooldown": 2.3,
				"armor_penetration": 6,
				"cost_wood": 60,
				"cost_gold": 40,
				"train_time": 42,
				"trained_at": "archery_range",
				"model": "res://rts_game/units/crossbowman.tscn",
				"icon": "res://rts_game/assets/icons/orc_crossbowman_icon.png"
			}
		},
		"buildings": {
			"archery_range": {
				"name": "War Lodge",
				"type": "military",
				"health": 550,
				"cost_wood": 140,
				"cost_gold": 60,
				"build_time": 55,
				"trains": ["archer", "crossbowman"],
				"model": "res://rts_game/buildings/archery_range.tscn",
				"icon": "res://rts_game/assets/icons/war_lodge_icon.png"
			}
		}
	},
	"elf": {
		"id": 2,
		"name": "Elven Alliance",
		"color": Color(0.2, 0.8, 0.4),
		"units": {
			"archer": {
				"name": "Elven Archer",
				"type": "ranged_infantry",
				"health": 55,
				"damage": 14,
				"range": 18,
				"speed": 4.0,
				"attack_cooldown": 1.3,
				"cost_wood": 55,
				"cost_gold": 30,
				"train_time": 32,
				"trained_at": "archery_range",
				"model": "res://rts_game/units/archer.tscn",
				"icon": "res://rts_game/assets/icons/elven_archer_icon.png"
			},
			"crossbowman": {
				"name": "Elven Marksman",
				"type": "ranged_infantry",
				"health": 60,
				"damage": 16,
				"range": 20,
				"speed": 3.5,
				"attack_cooldown": 2.0,
				"armor_penetration": 4,
				"cost_wood": 65,
				"cost_gold": 45,
				"train_time": 40,
				"trained_at": "archery_range",
				"model": "res://rts_game/units/crossbowman.tscn",
				"icon": "res://rts_game/assets/icons/elven_marksman_icon.png"
			}
		},
		"buildings": {
			"archery_range": {
				"name": "Hunter's Hall",
				"type": "military",
				"health": 450,
				"cost_wood": 160,
				"cost_gold": 40,
				"build_time": 50,
				"trains": ["archer", "crossbowman"],
				"model": "res://rts_game/buildings/archery_range.tscn",
				"icon": "res://rts_game/assets/icons/hunters_hall_icon.png"
			}
		}
	}
}

static func get_faction_data(faction_name: String) -> Dictionary:
	if FACTION_DATA.has(faction_name):
		return FACTION_DATA[faction_name]
	return {}

static func get_unit_data(faction_name: String, unit_name: String) -> Dictionary:
	var faction = get_faction_data(faction_name)
	if faction.has("units") and faction.units.has(unit_name):
		return faction.units[unit_name]
	return {}

static func get_building_data(faction_name: String, building_name: String) -> Dictionary:
	var faction = get_faction_data(faction_name)
	if faction.has("buildings") and faction.buildings.has(building_name):
		return faction.buildings[building_name]
	return {}

static func get_all_factions() -> Array:
	return FACTION_DATA.keys()

static func get_faction_color(faction_name: String) -> Color:
	var faction = get_faction_data(faction_name)
	if faction.has("color"):
		return faction.color
	return Color.WHITE
