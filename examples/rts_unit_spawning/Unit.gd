extends CharacterBody2D
class_name RTSUnit

## Basic RTS unit with movement capabilities

@export var move_speed := 100.0

var target_position: Vector2 = Vector2.ZERO
var is_moving := false


func _ready() -> void:
	# Start idle
	pass


func _physics_process(delta: float) -> void:
	if is_moving:
		var direction := (target_position - global_position).normalized()
		var distance := global_position.distance_to(target_position)
		
		if distance < 5.0:
			# Reached destination
			is_moving = false
			velocity = Vector2.ZERO
		else:
			velocity = direction * move_speed
		
		move_and_slide()


func move_to(world_position: Vector2) -> void:
	"""Command the unit to move to a specific position"""
	target_position = world_position
	is_moving = true
	print("Unit moving to: ", world_position)


func stop() -> void:
	"""Stop the unit from moving"""
	is_moving = false
	velocity = Vector2.ZERO


## Visual representation
func _draw() -> void:
	if Engine.is_editor_hint() or OS.is_debug_build():
		# Draw unit as a circle
		draw_circle(Vector2.ZERO, 10, Color.RED)
		
		# Draw movement target if moving
		if is_moving:
			var target_local := to_local(target_position)
			draw_line(Vector2.ZERO, target_local, Color.CYAN, 1.0, true)
