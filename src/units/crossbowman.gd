extends RangedUnit
class_name Crossbowman

## Crossbowman unit implementation
## Heavily armored ranged unit with high damage but slower rate of fire

func _ready():
	# Set crossbowman-specific configuration
	unit_id = "crossbowman"
	unit_name = "Crossbowman"
	faction = "human"
	
	# Base stats (higher damage, slower speed, more health than archer)
	max_health = 60.0
	current_health = max_health
	attack_damage = 12.0
	attack_range = 8.0
	move_speed = 3.0
	attack_speed = 0.5  # Slower than archer
	line_of_sight = 10.0
	
	# Projectile settings (faster, flatter trajectory)
	projectile_speed = 20.0
	projectile_arc = 0.15
	model_path = "res://models/units/crossbowman.glb"
	
	# Load from config and override if needed
	super._ready()

func _get_unit_type() -> String:
	return "crossbowman"

## Crossbowman-specific abilities could be added here
## For example: armor piercing, pavise shield deployment, etc.

func take_damage(amount: float, attacker: Node3D = null):
	"""Crossbowmen have higher armor, reduce incoming damage"""
	var armor_reduction = 0.2  # 20% damage reduction
	var reduced_damage = amount * (1.0 - armor_reduction)
	super.take_damage(reduced_damage, attacker)
