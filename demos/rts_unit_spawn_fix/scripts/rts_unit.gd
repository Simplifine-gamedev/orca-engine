extends CharacterBody3D
class_name RTSUnit

## RTS Unit with proper animation handling to prevent T-posing
## 
## This script demonstrates the fix for units appearing in T-pose when spawning.
## Key fix: Set animation state immediately in _ready() before first frame renders.

@export var move_speed: float = 5.0
@export var max_health: float = 100.0

var current_health: float = 100.0
var target_position: Vector3 = Vector3.ZERO
var is_moving: bool = false
var animation_player: AnimationPlayer
var animation_tree: AnimationTree
var state_machine: AnimationNodeStateMachinePlayback

enum UnitState {
	IDLE,
	MOVING,
	ATTACKING,
	DYING
}

var current_state: UnitState = UnitState.IDLE


func _ready() -> void:
	# CRITICAL FIX: Initialize animation components immediately
	_initialize_animation_system()
	
	# Set spawn position
	current_health = max_health
	
	# Force initial animation state BEFORE first physics frame
	# This prevents T-pose from showing during spawn
	_set_animation_state(UnitState.IDLE)
	
	# Ensure animation tree is active from spawn
	if animation_tree:
		animation_tree.active = true


func _initialize_animation_system() -> void:
	"""
	Initialize animation components and set up the state machine.
	This MUST be called before any animation state changes.
	"""
	# Find AnimationPlayer (usually child of the unit's visual mesh)
	animation_player = _find_animation_player(self)
	
	# Find AnimationTree if using state machine
	animation_tree = _find_animation_tree(self)
	
	if animation_tree:
		# Get the state machine playback for controlling transitions
		state_machine = animation_tree.get("parameters/playback")
		
		# IMPORTANT: Activate tree immediately to prevent T-pose
		animation_tree.active = true
		
		if state_machine:
			# Force immediate transition to idle (prevents T-pose)
			state_machine.travel("idle")
	elif animation_player:
		# Fallback: If no AnimationTree, play idle animation directly
		if animation_player.has_animation("idle"):
			animation_player.play("idle")
		elif animation_player.has_animation("Idle"):
			animation_player.play("Idle")


func _find_animation_player(node: Node) -> AnimationPlayer:
	"""Recursively find AnimationPlayer in children"""
	if node is AnimationPlayer:
		return node
	
	for child in node.get_children():
		var result = _find_animation_player(child)
		if result:
			return result
	
	return null


func _find_animation_tree(node: Node) -> AnimationTree:
	"""Recursively find AnimationTree in children"""
	if node is AnimationTree:
		return node
	
	for child in node.get_children():
		var result = _find_animation_tree(child)
		if result:
			return result
	
	return null


func _physics_process(delta: float) -> void:
	# Handle movement
	if is_moving and target_position:
		var direction = (target_position - global_position).normalized()
		var distance_to_target = global_position.distance_to(target_position)
		
		if distance_to_target > 0.1:
			velocity = direction * move_speed
			move_and_slide()
			
			# Update animation state
			if current_state != UnitState.MOVING:
				_set_animation_state(UnitState.MOVING)
		else:
			# Reached destination
			is_moving = false
			velocity = Vector3.ZERO
			_set_animation_state(UnitState.IDLE)
	else:
		velocity = Vector3.ZERO
		if current_state == UnitState.MOVING:
			_set_animation_state(UnitState.IDLE)


func _set_animation_state(new_state: UnitState) -> void:
	"""
	Safely change animation state.
	This ensures smooth transitions without T-posing.
	"""
	if current_state == new_state:
		return
	
	current_state = new_state
	
	# Use AnimationTree state machine if available
	if state_machine:
		match new_state:
			UnitState.IDLE:
				state_machine.travel("idle")
			UnitState.MOVING:
				state_machine.travel("move")
			UnitState.ATTACKING:
				state_machine.travel("attack")
			UnitState.DYING:
				state_machine.travel("death")
	
	# Fallback to direct AnimationPlayer control
	elif animation_player:
		match new_state:
			UnitState.IDLE:
				_play_animation("idle")
			UnitState.MOVING:
				_play_animation("move")
			UnitState.ATTACKING:
				_play_animation("attack")
			UnitState.DYING:
				_play_animation("death")


func _play_animation(anim_name: String) -> void:
	"""Play animation with fallback to capitalized names"""
	if not animation_player:
		return
	
	# Try lowercase first
	if animation_player.has_animation(anim_name):
		animation_player.play(anim_name)
	# Try capitalized version
	elif animation_player.has_animation(anim_name.capitalize()):
		animation_player.play(anim_name.capitalize())


func move_to(position: Vector3) -> void:
	"""Command unit to move to target position"""
	target_position = position
	is_moving = true


func take_damage(amount: float) -> void:
	"""Apply damage to unit"""
	current_health -= amount
	
	if current_health <= 0:
		die()


func die() -> void:
	"""Handle unit death"""
	_set_animation_state(UnitState.DYING)
	is_moving = false
	
	# Clean up after death animation completes
	if animation_player:
		await animation_player.animation_finished
	
	queue_free()


## Called by building when spawning this unit
func spawn_at_position(spawn_pos: Vector3) -> void:
	"""
	Properly spawn unit at given position.
	This is called by the building after instantiation.
	"""
	global_position = spawn_pos
	
	# Ensure idle animation plays immediately
	# (redundant with _ready() but safe for edge cases)
	if state_machine:
		state_machine.travel("idle")
	elif animation_player:
		_play_animation("idle")
