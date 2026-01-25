extends Node3D
class_name Projectile

## Generic projectile for ranged units
## Flies towards target with optional arc and applies damage on hit

signal hit_target(target)
signal projectile_destroyed

## Configuration
@export var damage: float = 8.0
@export var speed: float = 15.0
@export var arc_height: float = 0.3  # 0-1, how high the arc is
@export var rotate_with_direction: bool = true

## Effects
@export var trail_effect: PackedScene
@export var impact_effect: PackedScene
@export var hit_sound: AudioStream

## Internal state
var target: Node3D = null
var start_position: Vector3
var target_position: Vector3
var travel_time: float = 0.0
var max_travel_time: float = 2.0
var has_target: bool = false

func _ready():
	start_position = global_position
	
	# Add trail effect if available
	if trail_effect:
		var trail = trail_effect.instantiate()
		add_child(trail)

func set_target(new_target: Node3D, dmg: float = 8.0, spd: float = 15.0, arc: float = 0.3):
	"""Configure and launch the projectile"""
	target = new_target
	damage = dmg
	speed = spd
	arc_height = arc
	has_target = true
	
	if target:
		target_position = target.global_position
		var distance = start_position.distance_to(target_position)
		max_travel_time = distance / speed

func _physics_process(delta):
	if not has_target:
		return
	
	travel_time += delta
	
	# Update target position if target is still valid and moving
	if is_instance_valid(target):
		target_position = target.global_position
	
	# Calculate progress (0 to 1)
	var progress = travel_time / max_travel_time
	
	if progress >= 1.0:
		_on_reached_target()
		return
	
	# Linear interpolation for horizontal movement
	var horizontal_pos = start_position.lerp(target_position, progress)
	
	# Add arc to vertical position
	var arc_offset = sin(progress * PI) * arc_height * start_position.distance_to(target_position)
	global_position = horizontal_pos + Vector3(0, arc_offset, 0)
	
	# Rotate to face direction of travel
	if rotate_with_direction:
		var next_progress = min(progress + 0.01, 1.0)
		var next_pos = start_position.lerp(target_position, next_progress)
		var next_arc_offset = sin(next_progress * PI) * arc_height * start_position.distance_to(target_position)
		var next_position = next_pos + Vector3(0, next_arc_offset, 0)
		
		var direction = (next_position - global_position).normalized()
		if direction.length() > 0.01:
			look_at(global_position + direction, Vector3.UP)

func _on_reached_target():
	"""Handle reaching the target"""
	# Deal damage if target is still valid
	if is_instance_valid(target) and target.has_method("take_damage"):
		target.take_damage(damage, get_parent())
		hit_target.emit(target)
	
	# Spawn impact effect
	if impact_effect:
		var effect = impact_effect.instantiate()
		get_tree().current_scene.add_child(effect)
		effect.global_position = global_position
	
	# Play hit sound
	if hit_sound:
		var audio_player = AudioStreamPlayer3D.new()
		audio_player.stream = hit_sound
		get_tree().current_scene.add_child(audio_player)
		audio_player.global_position = global_position
		audio_player.play()
		audio_player.finished.connect(audio_player.queue_free)
	
	projectile_destroyed.emit()
	queue_free()

func _on_body_entered(body: Node):
	"""Handle collision with obstacles"""
	if body != target and not body.is_in_group("projectiles"):
		# Hit something else (wall, terrain, etc.)
		if impact_effect:
			var effect = impact_effect.instantiate()
			get_tree().current_scene.add_child(effect)
			effect.global_position = global_position
		
		projectile_destroyed.emit()
		queue_free()
