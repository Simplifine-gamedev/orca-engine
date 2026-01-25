extends Node2D
## RTS Building Unit Spawner Example
## 
## This script demonstrates how to properly handle unit spawning in RTS games.
## When a building has no rally point set, units spawn at a default position
## outside the building entrance instead of inside where they're hard to see.
##
## Usage: Attach this script to a building node in your RTS game.

class_name RTSBuilding

## The rally point where units should move after spawning (null = use default)
@export var rally_point: Vector2 = Vector2.ZERO
@export var has_rally_point: bool = false

## Default spawn offset from building center (in pixels)
## This should point to the building entrance/exit
@export var default_spawn_offset: Vector2 = Vector2(64, 32)

## Scene or class to spawn
@export var unit_scene: PackedScene

## Optional: Direction the building "faces" (for calculating spawn position)
@export_enum("Down", "Up", "Left", "Right") var building_facing: int = 0

## Queue of units waiting to spawn
var spawn_queue: Array[Dictionary] = []

## Time between spawns (in seconds)
@export var spawn_interval: float = 1.0
var spawn_timer: float = 0.0


func _ready() -> void:
	# Initialize rally point if not set
	if not has_rally_point:
		rally_point = global_position + default_spawn_offset


func _process(delta: float) -> void:
	# Handle spawn queue
	if spawn_queue.size() > 0:
		spawn_timer += delta
		if spawn_timer >= spawn_interval:
			spawn_timer = 0.0
			_spawn_next_unit()


## Add a unit to the spawn queue
func queue_unit_spawn(unit_type: String = "default") -> void:
	spawn_queue.append({
		"type": unit_type,
		"queued_at": Time.get_ticks_msec()
	})


## Set the rally point for spawned units
func set_rally_point(point: Vector2) -> void:
	rally_point = point
	has_rally_point = true


## Clear the rally point (revert to default spawn position)
func clear_rally_point() -> void:
	has_rally_point = false
	rally_point = global_position + default_spawn_offset


## Get the spawn position (default position outside building)
func get_spawn_position() -> Vector2:
	# Calculate spawn position based on building facing direction
	var spawn_pos: Vector2 = global_position
	
	match building_facing:
		0: # Down
			spawn_pos += Vector2(0, default_spawn_offset.y)
		1: # Up
			spawn_pos += Vector2(0, -default_spawn_offset.y)
		2: # Left
			spawn_pos += Vector2(-default_spawn_offset.x, 0)
		3: # Right
			spawn_pos += Vector2(default_spawn_offset.x, 0)
		_:
			spawn_pos += default_spawn_offset
	
	return spawn_pos


## Get the rally point (or default spawn position if no rally point set)
func get_rally_point() -> Vector2:
	if has_rally_point:
		return rally_point
	else:
		# Return default position outside building
		return get_spawn_position()


## Spawn the next unit in the queue
func _spawn_next_unit() -> void:
	if spawn_queue.is_empty():
		return
	
	var unit_data = spawn_queue.pop_front()
	
	# Create the unit instance
	if not unit_scene:
		push_error("RTSBuilding: No unit_scene assigned!")
		return
	
	var unit = unit_scene.instantiate()
	
	# Set spawn position (OUTSIDE the building)
	var spawn_pos = get_spawn_position()
	unit.global_position = spawn_pos
	
	# Add to scene tree
	get_parent().add_child(unit)
	
	# Send unit to rally point if it exists, or let it stay at spawn position
	if has_rally_point:
		# Assuming unit has a move_to method
		if unit.has_method("move_to"):
			unit.move_to(rally_point)
	
	# Emit signal for UI updates
	unit_spawned.emit(unit, spawn_pos, get_rally_point())


## Signal emitted when a unit is spawned
signal unit_spawned(unit: Node, spawn_pos: Vector2, rally_point: Vector2)


## Example: Get spawn position for visualization (e.g., for spawn effects)
func get_spawn_visualization_position() -> Vector2:
	return get_spawn_position()


## Debug helper: Draw spawn position and rally point
func _draw() -> void:
	if Engine.is_editor_hint():
		return
	
	# Draw default spawn position (green circle)
	var spawn_pos = get_spawn_position() - global_position
	draw_circle(spawn_pos, 8, Color.GREEN)
	
	# Draw rally point if set (blue circle and line)
	if has_rally_point:
		var rally_local = rally_point - global_position
		draw_circle(rally_local, 10, Color.BLUE)
		draw_line(spawn_pos, rally_local, Color.CYAN, 2.0)


## Example usage function for testing
func _example_usage() -> void:
	# Spawn a unit without rally point (will spawn at default position outside building)
	queue_unit_spawn("worker")
	
	# Set a rally point
	set_rally_point(Vector2(200, 200))
	
	# Spawn a unit with rally point (will spawn outside, then move to rally point)
	queue_unit_spawn("soldier")
	
	# Clear rally point (subsequent units spawn at default position)
	clear_rally_point()
