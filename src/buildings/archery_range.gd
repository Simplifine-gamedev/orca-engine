extends Node3D
class_name ArcheryRange

## Archery Range building
## Trains archers and crossbowmen

signal unit_training_started(unit_type: String)
signal unit_training_completed(unit_type: String, unit_instance: Node3D)
signal unit_training_cancelled(unit_type: String)

## Building configuration
@export var building_id: String = "archery_range"
@export var building_name: String = "Archery Range"
@export var faction: String = "human"

## Stats
@export_group("Stats")
@export var max_health: float = 800.0
@export var current_health: float = 800.0
@export var defense: float = 5.0

## Model
@export_group("Model")
@export var model_path: String = "res://models/buildings/archery_range.glb"

## Training
@export_group("Training")
@export var archer_scene: PackedScene
@export var crossbowman_scene: PackedScene
@export var unit_spawn_point: Node3D
@export var rally_point: Node3D

## Training queue
var training_queue: Array[Dictionary] = []
var current_training: Dictionary = {}
var is_training: bool = false

## Unit costs (gold, wood, food)
const UNIT_COSTS = {
	"archer": {"gold": 40, "wood": 25, "food": 1, "time": 20},
	"crossbowman": {"gold": 60, "wood": 40, "food": 1, "time": 30}
}

func _ready():
	current_health = max_health
	_load_building_config()

func _process(delta):
	if is_training:
		_update_training(delta)

func _load_building_config():
	"""Load building configuration from JSON"""
	var config_path = "res://generated_factions/all_factions_characters.json"
	if FileAccess.file_exists(config_path):
		var file = FileAccess.open(config_path, FileAccess.READ)
		if file:
			var json = JSON.new()
			var parse_result = json.parse(file.get_as_text())
			if parse_result == OK:
				var data = json.data
				if data.has("buildings") and data.buildings.has(building_id):
					var building_data = data.buildings[building_id]
					_apply_building_config(building_data)
			file.close()

func _apply_building_config(config: Dictionary):
	"""Apply configuration from JSON data"""
	building_name = config.get("name", building_name)
	faction = config.get("faction", faction)
	
	if config.has("stats"):
		var stats = config.stats
		max_health = stats.get("health", max_health)
		current_health = max_health
		defense = stats.get("defense", defense)

func train_unit(unit_type: String) -> bool:
	"""Add a unit to the training queue"""
	if not UNIT_COSTS.has(unit_type):
		push_error("Unknown unit type: " + unit_type)
		return false
	
	var cost = UNIT_COSTS[unit_type]
	
	# Check resources (this would connect to a resource manager)
	if not _has_resources(cost):
		return false
	
	# Deduct resources
	_deduct_resources(cost)
	
	# Add to queue
	var training_data = {
		"type": unit_type,
		"time_remaining": cost.time,
		"total_time": cost.time
	}
	
	training_queue.append(training_data)
	
	# Start training if not already training
	if not is_training:
		_start_next_training()
	
	return true

func cancel_training(index: int = 0):
	"""Cancel unit training and refund resources"""
	if index < 0 or index >= training_queue.size():
		return
	
	var training_data = training_queue[index]
	var unit_type = training_data.type
	var cost = UNIT_COSTS[unit_type]
	
	# Refund resources (partial refund based on progress)
	var progress = 1.0 - (training_data.time_remaining / training_data.total_time)
	var refund = {}
	for resource in cost:
		if resource != "time":
			refund[resource] = cost[resource] * (1.0 - progress * 0.5)  # 50% of progress lost
	
	_refund_resources(refund)
	
	# Remove from queue
	training_queue.remove_at(index)
	
	unit_training_cancelled.emit(unit_type)
	
	# If was current training, start next
	if index == 0 and is_training:
		is_training = false
		current_training = {}
		_start_next_training()

func _start_next_training():
	"""Start training the next unit in queue"""
	if training_queue.is_empty():
		is_training = false
		current_training = {}
		return
	
	current_training = training_queue[0]
	is_training = true
	
	unit_training_started.emit(current_training.type)

func _update_training(delta: float):
	"""Update current training progress"""
	if current_training.is_empty():
		return
	
	current_training.time_remaining -= delta
	
	if current_training.time_remaining <= 0:
		_complete_training()

func _complete_training():
	"""Complete current unit training"""
	var unit_type = current_training.type
	
	# Spawn unit
	var unit_instance = _spawn_unit(unit_type)
	
	if unit_instance:
		unit_training_completed.emit(unit_type, unit_instance)
	
	# Remove from queue
	training_queue.remove_at(0)
	current_training = {}
	is_training = false
	
	# Start next in queue
	_start_next_training()

func _spawn_unit(unit_type: String) -> Node3D:
	"""Spawn a trained unit"""
	var unit_scene: PackedScene = null
	
	match unit_type:
		"archer":
			unit_scene = archer_scene
		"crossbowman":
			unit_scene = crossbowman_scene
	
	if not unit_scene:
		push_error("No scene configured for unit type: " + unit_type)
		return null
	
	var unit_instance = unit_scene.instantiate()
	get_tree().current_scene.add_child(unit_instance)
	
	# Position at spawn point
	var spawn_pos = unit_spawn_point.global_position if unit_spawn_point else global_position
	unit_instance.global_position = spawn_pos
	
	# Send to rally point if set
	if rally_point and unit_instance.has_method("move_to"):
		unit_instance.move_to(rally_point.global_position)
	
	return unit_instance

func set_rally_point(position: Vector3):
	"""Set the rally point for trained units"""
	if not rally_point:
		rally_point = Node3D.new()
		add_child(rally_point)
	
	rally_point.global_position = position

func take_damage(amount: float, attacker: Node3D = null):
	"""Take damage from an attack"""
	var actual_damage = max(amount - defense, amount * 0.1)  # Min 10% damage
	current_health -= actual_damage
	
	if current_health <= 0:
		destroy()

func destroy():
	"""Destroy the building"""
	# Cancel all training
	while not training_queue.is_empty():
		cancel_training(0)
	
	# Play destruction effect/animation
	queue_free()

func get_health_percentage() -> float:
	"""Get health as percentage"""
	return current_health / max_health if max_health > 0 else 0.0

func get_training_progress() -> float:
	"""Get current training progress (0-1)"""
	if not is_training or current_training.is_empty():
		return 0.0
	
	var progress = 1.0 - (current_training.time_remaining / current_training.total_time)
	return clamp(progress, 0.0, 1.0)

func get_queue_size() -> int:
	"""Get number of units in training queue"""
	return training_queue.size()

## Resource management helpers (would connect to actual resource system)
func _has_resources(cost: Dictionary) -> bool:
	# TODO: Connect to game resource manager
	return true

func _deduct_resources(cost: Dictionary):
	# TODO: Connect to game resource manager
	pass

func _refund_resources(refund: Dictionary):
	# TODO: Connect to game resource manager
	pass
