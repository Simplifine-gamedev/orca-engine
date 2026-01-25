extends RangedUnit
class_name Archer

## Archer unit implementation
## Fast-moving ranged unit with moderate damage and good rate of fire

func _ready():
	# Set archer-specific configuration
	unit_id = "archer"
	unit_name = "Archer"
	faction = "human"
	
	# Base stats
	max_health = 50.0
	current_health = max_health
	attack_damage = 8.0
	attack_range = 7.0
	move_speed = 3.5
	attack_speed = 0.8
	line_of_sight = 9.0
	
	# Projectile settings
	projectile_speed = 15.0
	projectile_arc = 0.3
	model_path = "res://models/units/archer.glb"
	
	# Load from config and override if needed
	super._ready()

func _get_unit_type() -> String:
	return "archer"

## Archer-specific abilities could be added here
## For example: volley fire, fire arrows, etc.
