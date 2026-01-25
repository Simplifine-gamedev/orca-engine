extends Node
class_name FactionConfig

## Faction Configuration
## Defines building models and properties for each faction

# Faction types
enum Faction {
	HUMAN,
	DWARF,
	ELF,
	UNDEAD
}

# Faction names
const FACTION_NAMES = {
	Faction.HUMAN: "Human",
	Faction.DWARF: "Dwarf",
	Faction.ELF: "Elf",
	Faction.UNDEAD: "Undead"
}

# Building types
const BARRACKS = "barracks"
const TOWN_HALL = "town_hall"
const FARM = "farm"

# Building models for each faction
# In a real game, these would be paths to 3D models or scenes
const BUILDING_MODELS = {
	"human": {
		"barracks": "res://models/human_barracks.tscn",
		"town_hall": "res://models/human_town_hall.tscn",
		"farm": "res://models/human_farm.tscn"
	},
	"dwarf": {
		"barracks": "res://models/dwarf_barracks.tscn",
		"town_hall": "res://models/dwarf_town_hall.tscn",
		"farm": "res://models/dwarf_farm.tscn"
	},
	"elf": {
		"barracks": "res://models/elf_barracks.tscn",
		"town_hall": "res://models/elf_town_hall.tscn",
		"farm": "res://models/elf_farm.tscn"
	},
	"undead": {
		"barracks": "res://models/undead_barracks.tscn",
		"town_hall": "res://models/undead_town_hall.tscn",
		"farm": "res://models/undead_farm.tscn"
	}
}

# Building colors for visual representation (used in demo)
const BUILDING_COLORS = {
	"human": {
		"barracks": Color(0.8, 0.6, 0.4),  # Brown
		"town_hall": Color(0.7, 0.7, 0.9),  # Light blue
		"farm": Color(0.9, 0.8, 0.5)  # Yellow
	},
	"dwarf": {
		"barracks": Color(0.5, 0.5, 0.5),  # Gray (stone)
		"town_hall": Color(0.4, 0.4, 0.5),  # Dark gray
		"farm": Color(0.6, 0.5, 0.4)  # Dark brown
	},
	"elf": {
		"barracks": Color(0.5, 0.8, 0.5),  # Green
		"town_hall": Color(0.8, 0.9, 0.7),  # Light green
		"farm": Color(0.7, 0.9, 0.6)  # Bright green
	},
	"undead": {
		"barracks": Color(0.3, 0.2, 0.3),  # Dark purple
		"town_hall": Color(0.2, 0.2, 0.2),  # Black
		"farm": Color(0.4, 0.3, 0.3)  # Dark brown
	}
}

# Get faction name as string (for dictionary lookups)
static func get_faction_key(faction: Faction) -> String:
	match faction:
		Faction.HUMAN:
			return "human"
		Faction.DWARF:
			return "dwarf"
		Faction.ELF:
			return "elf"
		Faction.UNDEAD:
			return "undead"
		_:
			return "human"  # Default fallback

# Get building color for a faction and building type
static func get_building_color(faction: Faction, building_type: String) -> Color:
	var faction_key = get_faction_key(faction)
	if faction_key in BUILDING_COLORS and building_type in BUILDING_COLORS[faction_key]:
		return BUILDING_COLORS[faction_key][building_type]
	return Color.WHITE  # Fallback

# Get building model path for a faction and building type
static func get_building_model(faction: Faction, building_type: String) -> String:
	var faction_key = get_faction_key(faction)
	if faction_key in BUILDING_MODELS and building_type in BUILDING_MODELS[faction_key]:
		return BUILDING_MODELS[faction_key][building_type]
	return ""  # Fallback
