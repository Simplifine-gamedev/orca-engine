extends StaticBody3D
class_name ArcheryRange

# Archery Range building
# Trains archer and crossbowman units

@export var max_health: float = 500.0
@export var archer_train_time: float = 30.0
@export var crossbowman_train_time: float = 45.0
@export var archer_cost_wood: int = 50
@export var archer_cost_gold: int = 25
@export var crossbowman_cost_wood: int = 60
@export var crossbowman_cost_gold: int = 40

var current_health: float
var faction_id: int = 0
var training_queue: Array = []
var current_training: Dictionary = {}
var rally_point: Vector3

@onready var model: Node3D = $ArcheryRangeModel
@onready var spawn_point: Marker3D = $SpawnPoint
@onready var training_progress_bar: ProgressBar = $UI/TrainingProgress

# Unit types that can be trained
enum UnitType {
	ARCHER,
	CROSSBOWMAN
}

signal unit_trained(unit: Node3D)
signal training_started(unit_type: UnitType)
signal building_destroyed()

func _ready():
	current_health = max_health
	rally_point = global_position + Vector3(3, 0, 0)
	setup_building()

func setup_building():
	# Setup building model and UI
	if training_progress_bar:
		training_progress_bar.visible = false

func _process(delta):
	if current_training.size() > 0:
		update_training(delta)

func update_training(delta):
	current_training.time_remaining -= delta
	
	if training_progress_bar:
		var progress = 1.0 - (current_training.time_remaining / current_training.total_time)
		training_progress_bar.value = progress * 100
		training_progress_bar.visible = true
	
	if current_training.time_remaining <= 0:
		complete_training()

func complete_training():
	var unit_type = current_training.unit_type
	spawn_unit(unit_type)
	
	current_training.clear()
	if training_progress_bar:
		training_progress_bar.visible = false
	
	# Start next in queue
	if training_queue.size() > 0:
		var next_training = training_queue.pop_front()
		start_training(next_training)

func spawn_unit(unit_type: UnitType):
	var unit_scene: PackedScene
	
	match unit_type:
		UnitType.ARCHER:
			unit_scene = preload("res://rts_game/units/archer.tscn")
		UnitType.CROSSBOWMAN:
			unit_scene = preload("res://rts_game/units/crossbowman.tscn")
	
	if unit_scene:
		var unit = unit_scene.instantiate()
		get_tree().root.add_child(unit)
		
		if spawn_point:
			unit.global_position = spawn_point.global_position
		else:
			unit.global_position = global_position + Vector3(2, 0, 0)
		
		unit.set_faction(faction_id)
		unit_trained.emit(unit)
		
		# Move unit to rally point
		if unit.has_method("move_to"):
			unit.move_to(rally_point)

func train_archer() -> bool:
	# Check if player has resources (handled externally)
	# This method assumes resources are already checked
	var training_data = {
		"unit_type": UnitType.ARCHER,
		"total_time": archer_train_time,
		"time_remaining": archer_train_time
	}
	
	if current_training.size() == 0:
		start_training(training_data)
	else:
		training_queue.append(training_data)
	
	training_started.emit(UnitType.ARCHER)
	return true

func train_crossbowman() -> bool:
	# Check if player has resources (handled externally)
	# This method assumes resources are already checked
	var training_data = {
		"unit_type": UnitType.CROSSBOWMAN,
		"total_time": crossbowman_train_time,
		"time_remaining": crossbowman_train_time
	}
	
	if current_training.size() == 0:
		start_training(training_data)
	else:
		training_queue.append(training_data)
	
	training_started.emit(UnitType.CROSSBOWMAN)
	return true

func start_training(training_data: Dictionary):
	current_training = training_data.duplicate()

func set_rally_point(position: Vector3):
	rally_point = position

func take_damage(amount: float):
	current_health -= amount
	if current_health <= 0:
		destroy_building()

func destroy_building():
	building_destroyed.emit()
	queue_free()

func set_faction(id: int):
	faction_id = id

func get_faction() -> int:
	return faction_id

func cancel_training():
	if current_training.size() > 0:
		# Refund resources (handled externally)
		current_training.clear()
		if training_progress_bar:
			training_progress_bar.visible = false
		
		# Start next in queue
		if training_queue.size() > 0:
			var next_training = training_queue.pop_front()
			start_training(next_training)

func get_training_queue_size() -> int:
	var size = training_queue.size()
	if current_training.size() > 0:
		size += 1
	return size
