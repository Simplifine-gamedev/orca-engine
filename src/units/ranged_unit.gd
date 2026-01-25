extends CharacterBody3D
class_name RangedUnit

## Base class for ranged units (Archer, Crossbowman, etc.)
## Handles movement, targeting, and ranged attacks with projectiles

signal unit_died
signal target_acquired(target)
signal projectile_fired(projectile)

## Unit configuration
@export var unit_id: String = "archer"
@export var unit_name: String = "Archer"
@export var faction: String = "human"

## Stats
@export_group("Stats")
@export var max_health: float = 50.0
@export var current_health: float = 50.0
@export var attack_damage: float = 8.0
@export var attack_range: float = 7.0
@export var move_speed: float = 3.5
@export var attack_speed: float = 0.8  # attacks per second
@export var line_of_sight: float = 9.0

## Model and Animation
@export_group("Model")
@export var model_path: String = "res://models/units/archer.glb"
@export var animation_player: AnimationPlayer

## Projectile settings
@export_group("Projectile")
@export var projectile_scene: PackedScene
@export var projectile_spawn_point: Node3D
@export var projectile_speed: float = 15.0
@export var projectile_arc: float = 0.3

## Internal state
var current_target: Node3D = null
var is_attacking: bool = false
var can_attack: bool = true
var attack_cooldown: float = 0.0

## Movement
var move_target: Vector3 = Vector3.ZERO
var has_move_target: bool = false

func _ready():
	# Load unit configuration if available
	_load_unit_config()
	
	# Initialize
	current_health = max_health
	attack_cooldown = 1.0 / attack_speed

func _physics_process(delta):
	# Handle attacking
	if is_attacking and current_target:
		_face_target(current_target.global_position)
		_try_attack(delta)
	
	# Handle movement
	elif has_move_target:
		_move_to_target(delta)
	
	# Update attack cooldown
	if not can_attack:
		attack_cooldown -= delta
		if attack_cooldown <= 0:
			can_attack = true
			attack_cooldown = 1.0 / attack_speed

func _load_unit_config():
	"""Load unit configuration from JSON"""
	var config_path = "res://generated_factions/all_factions_characters.json"
	if FileAccess.file_exists(config_path):
		var file = FileAccess.open(config_path, FileAccess.READ)
		if file:
			var json = JSON.new()
			var parse_result = json.parse(file.get_as_text())
			if parse_result == OK:
				var data = json.data
				if data.has("characters") and data.characters.has(unit_id):
					var unit_data = data.characters[unit_id]
					_apply_unit_config(unit_data)
			file.close()

func _apply_unit_config(config: Dictionary):
	"""Apply configuration from JSON data"""
	unit_name = config.get("name", unit_name)
	faction = config.get("faction", faction)
	
	if config.has("stats"):
		var stats = config.stats
		max_health = stats.get("health", max_health)
		current_health = max_health
		attack_damage = stats.get("attack", attack_damage)
		attack_range = stats.get("attackRange", attack_range)
		move_speed = stats.get("moveSpeed", move_speed)
		attack_speed = stats.get("attackSpeed", attack_speed)
		line_of_sight = stats.get("lineOfSight", line_of_sight)
	
	if config.has("projectile"):
		var proj = config.projectile
		projectile_speed = proj.get("speed", projectile_speed)
		projectile_arc = proj.get("arc", projectile_arc)

func set_target(target: Node3D):
	"""Set attack target"""
	current_target = target
	is_attacking = true
	has_move_target = false
	target_acquired.emit(target)

func move_to(position: Vector3):
	"""Move to position"""
	move_target = position
	has_move_target = true
	is_attacking = false
	current_target = null
	
	if animation_player:
		animation_player.play("walk")

func stop():
	"""Stop all actions"""
	is_attacking = false
	has_move_target = false
	current_target = null
	
	if animation_player:
		animation_player.play("idle")

func _move_to_target(delta: float):
	"""Move towards target position"""
	var direction = (move_target - global_position).normalized()
	var distance = global_position.distance_to(move_target)
	
	if distance < 0.5:
		has_move_target = false
		stop()
		return
	
	velocity = direction * move_speed
	move_and_slide()
	
	# Face movement direction
	if direction.length() > 0.01:
		var look_at_pos = global_position + direction
		look_at(look_at_pos, Vector3.UP)

func _face_target(target_pos: Vector3):
	"""Face the target"""
	var direction = (target_pos - global_position).normalized()
	if direction.length() > 0.01:
		var look_at_pos = global_position + direction
		look_at(look_at_pos, Vector3.UP)

func _try_attack(delta: float):
	"""Try to attack the current target"""
	if not current_target or not can_attack:
		return
	
	var distance = global_position.distance_to(current_target.global_position)
	
	if distance <= attack_range:
		_fire_projectile()
		can_attack = false
		
		if animation_player:
			animation_player.play("attack")

func _fire_projectile():
	"""Fire a projectile at the current target"""
	if not projectile_scene or not current_target:
		return
	
	var projectile = projectile_scene.instantiate()
	
	# Get spawn point or use unit position
	var spawn_pos = projectile_spawn_point.global_position if projectile_spawn_point else global_position + Vector3(0, 1.5, 0)
	
	get_tree().current_scene.add_child(projectile)
	projectile.global_position = spawn_pos
	
	# Configure projectile
	if projectile.has_method("set_target"):
		projectile.set_target(current_target, attack_damage, projectile_speed, projectile_arc)
	
	projectile_fired.emit(projectile)

func take_damage(amount: float, attacker: Node3D = null):
	"""Take damage from an attack"""
	current_health -= amount
	
	if current_health <= 0:
		die()
	
	# Optional: Fight back
	if attacker and not current_target:
		set_target(attacker)

func die():
	"""Handle unit death"""
	is_attacking = false
	has_move_target = false
	
	if animation_player:
		animation_player.play("death")
	
	unit_died.emit()
	
	# Wait for death animation then remove
	await get_tree().create_timer(2.0).timeout
	queue_free()

func heal(amount: float):
	"""Heal the unit"""
	current_health = min(current_health + amount, max_health)

func get_health_percentage() -> float:
	"""Get health as percentage"""
	return current_health / max_health if max_health > 0 else 0.0
