extends CharacterBody2D
## RTS Unit Example
## 
## Simple unit script that works with the RTSBuilding spawner.
## Demonstrates proper unit movement after spawning.

class_name RTSUnit

## Movement speed in pixels per second
@export var move_speed: float = 100.0

## Target position to move to
var target_position: Vector2 = Vector2.ZERO
var has_target: bool = false

## Selection state
var is_selected: bool = false


func _ready() -> void:
	# Unit starts at spawn position, not inside building
	pass


func _physics_process(delta: float) -> void:
	if has_target:
		_move_towards_target(delta)


## Move unit to a specific position (called by building or player command)
func move_to(pos: Vector2) -> void:
	target_position = pos
	has_target = true


## Stop moving
func stop() -> void:
	has_target = false
	velocity = Vector2.ZERO


func _move_towards_target(delta: float) -> void:
	var direction = (target_position - global_position).normalized()
	var distance = global_position.distance_to(target_position)
	
	if distance < 5.0:
		# Reached target
		has_target = false
		velocity = Vector2.ZERO
		global_position = target_position
	else:
		# Move towards target
		velocity = direction * move_speed
		move_and_slide()


func _draw() -> void:
	# Draw unit (simple circle for example)
	draw_circle(Vector2.ZERO, 8, Color.WHITE if not is_selected else Color.YELLOW)
	
	# Draw target indicator if moving
	if has_target:
		var target_local = target_position - global_position
		draw_line(Vector2.ZERO, target_local, Color.GREEN, 1.0)


func select() -> void:
	is_selected = true
	queue_redraw()


func deselect() -> void:
	is_selected = false
	queue_redraw()
