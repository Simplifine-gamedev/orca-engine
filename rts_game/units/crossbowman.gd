extends CharacterBody3D
class_name Crossbowman

# Crossbowman unit with powerful ranged crossbow attack
# Trained from Archery Range building
# Stronger than archer but slower reload

@export var max_health: float = 70.0
@export var movement_speed: float = 3.0
@export var attack_damage: float = 18.0
@export var attack_range: float = 18.0
@export var attack_cooldown: float = 2.2
@export var projectile_speed: float = 25.0
@export var armor_penetration: float = 5.0

var current_health: float
var current_target: Node3D = null
var attack_timer: float = 0.0
var faction_id: int = 0
var is_reloading: bool = false

# Animation states
enum AnimationState {
	IDLE,
	WALKING,
	ATTACKING,
	RELOADING,
	DYING
}

var current_animation_state: AnimationState = AnimationState.IDLE

@onready var animation_player: AnimationPlayer = $AnimationPlayer
@onready var model: Node3D = $CrossbowmanModel
@onready var attack_point: Node3D = $AttackPoint

func _ready():
	current_health = max_health
	setup_model()

func setup_model():
	# Setup 3D model and animations
	# Model should have: idle, walk, attack_crossbow, reload, and death animations
	if animation_player:
		animation_player.animation_finished.connect(_on_animation_finished)

func _physics_process(delta):
	if attack_timer > 0:
		attack_timer -= delta
	
	if current_target and is_instance_valid(current_target):
		handle_combat(delta)
	else:
		if not is_reloading:
			current_animation_state = AnimationState.IDLE
			if animation_player and animation_player.current_animation != "idle":
				animation_player.play("idle")

func handle_combat(delta):
	var distance = global_position.distance_to(current_target.global_position)
	
	if distance > attack_range:
		# Move towards target
		if not is_reloading:
			move_towards_target(delta)
	else:
		# In range, attack
		if not is_reloading:
			look_at_target()
			if attack_timer <= 0:
				perform_attack()
				attack_timer = attack_cooldown

func move_towards_target(delta):
	var direction = (current_target.global_position - global_position).normalized()
	velocity = direction * movement_speed
	move_and_slide()
	
	look_at(current_target.global_position)
	rotation.x = 0  # Keep upright
	rotation.z = 0
	
	if current_animation_state != AnimationState.WALKING:
		current_animation_state = AnimationState.WALKING
		if animation_player:
			animation_player.play("walk")

func look_at_target():
	if current_target:
		look_at(current_target.global_position)
		rotation.x = 0
		rotation.z = 0

func perform_attack():
	current_animation_state = AnimationState.ATTACKING
	if animation_player:
		animation_player.play("attack_crossbow")
	
	# Spawn crossbow bolt projectile
	spawn_projectile()
	
	# Start reload animation after attack
	is_reloading = true

func spawn_projectile():
	var projectile = preload("res://rts_game/projectiles/crossbow_bolt.tscn").instantiate()
	get_tree().root.add_child(projectile)
	
	if attack_point:
		projectile.global_position = attack_point.global_position
	else:
		projectile.global_position = global_position + Vector3(0, 1.5, 0)
	
	if current_target:
		var direction = (current_target.global_position - projectile.global_position).normalized()
		projectile.set_direction(direction, projectile_speed, attack_damage, faction_id, armor_penetration)

func take_damage(amount: float):
	current_health -= amount
	if current_health <= 0:
		die()

func die():
	current_animation_state = AnimationState.DYING
	if animation_player:
		animation_player.play("death")
	set_physics_process(false)
	# Will be freed after death animation

func set_target(target: Node3D):
	current_target = target

func set_faction(id: int):
	faction_id = id

func _on_animation_finished(anim_name: String):
	if anim_name == "death":
		queue_free()
	elif anim_name == "attack_crossbow":
		# Start reload animation
		current_animation_state = AnimationState.RELOADING
		if animation_player:
			animation_player.play("reload")
	elif anim_name == "reload":
		is_reloading = false
		current_animation_state = AnimationState.IDLE
