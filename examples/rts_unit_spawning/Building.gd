extends Node2D
class_name RTSBuilding

## Handles unit spawning for RTS buildings with rally point support
## This solves the issue where units spawn inside buildings when no rally point is set

@export var spawn_scene: PackedScene ## The unit to spawn
@export var default_spawn_offset := Vector2(100, 0) ## Default offset from building center when no rally point is set
@export var spawn_interval := 2.0 ## Time between spawns in seconds

var rally_point: Vector2 = Vector2.ZERO ## Position where units should move after spawning (world coordinates)
var has_rally_point := false ## Whether a rally point has been set

var spawn_timer: Timer


func _ready() -> void:
	setup_spawn_timer()


func setup_spawn_timer() -> void:
	spawn_timer = Timer.new()
	spawn_timer.wait_time = spawn_interval
	spawn_timer.timeout.connect(_on_spawn_timer_timeout)
	add_child(spawn_timer)


func start_production() -> void:
	"""Start producing units"""
	if spawn_scene and not spawn_timer.is_stopped():
		spawn_timer.start()


func stop_production() -> void:
	"""Stop producing units"""
	spawn_timer.stop()


func set_rally_point(world_position: Vector2) -> void:
	"""Set the rally point where spawned units should move to"""
	rally_point = world_position
	has_rally_point = true
	print("Rally point set to: ", rally_point)


func clear_rally_point() -> void:
	"""Clear the rally point"""
	has_rally_point = false
	print("Rally point cleared")


func get_spawn_position() -> Vector2:
	"""
	Get the position where units should spawn.
	Returns a position outside the building to avoid units spawning inside.
	
	BUG FIX: This ensures units always spawn at a visible location,
	even when no rally point is set.
	"""
	if has_rally_point:
		# If rally point is set, spawn slightly towards it from the building
		var direction_to_rally := (rally_point - global_position).normalized()
		return global_position + direction_to_rally * default_spawn_offset.length()
	else:
		# DEFAULT BEHAVIOR: Spawn at a fixed offset from building entrance
		# This fixes the bug where units spawn inside the building
		return global_position + default_spawn_offset


func spawn_unit() -> Node:
	"""
	Spawn a unit at the appropriate location.
	Units will spawn outside the building, not inside it.
	"""
	if not spawn_scene:
		push_error("No spawn scene set for building")
		return null
	
	var unit = spawn_scene.instantiate()
	get_parent().add_child(unit)
	
	# Set spawn position (outside the building)
	var spawn_pos := get_spawn_position()
	unit.global_position = spawn_pos
	
	# If there's a rally point and unit has move command, send it there
	if has_rally_point and unit.has_method("move_to"):
		unit.move_to(rally_point)
	
	print("Unit spawned at: ", spawn_pos)
	return unit


func _on_spawn_timer_timeout() -> void:
	spawn_unit()


## Visual feedback for debugging
func _draw() -> void:
	if Engine.is_editor_hint() or OS.is_debug_build():
		# Draw building bounds
		draw_rect(Rect2(-32, -32, 64, 64), Color.BLUE, false, 2.0)
		
		# Draw default spawn location
		var spawn_offset := get_spawn_position() - global_position
		draw_circle(spawn_offset, 5, Color.GREEN)
		draw_line(Vector2.ZERO, spawn_offset, Color.GREEN, 1.0)
		
		# Draw rally point if set
		if has_rally_point:
			var rally_local := to_local(rally_point)
			draw_circle(rally_local, 8, Color.YELLOW)
			draw_line(spawn_offset, rally_local, Color.YELLOW, 2.0, true)
