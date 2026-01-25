extends CharacterBody3D
class_name Archer

# Archer unit with ranged bow attack
# Trained from Archery Range building

@export var max_health: float = 60.0
@export var movement_speed: float = 3.5
@export var attack_damage: float = 12.0
@export var attack_range: float = 15.0
@export var attack_cooldown: float = 1.5
@export var projectile_speed: float = 20.0

var current_health: float
var current_target: Node3D = null
var attack_timer: float = 0.0
var faction_id: int = 0

# Animation states
enum AnimationState {
	IDLE,
	WALKING,
	ATTACKING,
	DYING
}

var current_animation_state: AnimationState = AnimationState.IDLE

@onready var animation_player: AnimationPlayer = $AnimationPlayer
@onready var model: Node3D = $ArcherModel
@onready var attack_point: Node3D = $AttackPoint

func _ready():
	current_health = max_health
	setup_model()

func setup_model():
	# Setup 3D model and animations
	# Model should have: idle, walk, attack_bow, and death animations
	if animation_player:
		animation_player.animation_finished.connect(_on_animation_finished)

func _physics_process(delta):
	if attack_timer > 0:
		attack_timer -= delta
	
	if current_target and is_instance_valid(current_target):
		handle_combat(delta)
	else:
		current_animation_state = AnimationState.IDLE
		if animation_player and animation_player.current_animation != "idle":
			animation_player.play("idle")

func handle_combat(delta):
	var distance = global_position.distance_to(current_target.global_position)
	
	if distance > attack_range:
		# Move towards target
		move_towards_target(delta)
	else:
		# In range, attack
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
		animation_player.play("attack_bow")
	
	# Spawn arrow projectile
	spawn_projectile()

func spawn_projectile():
	var projectile = preload("res://rts_game/projectiles/arrow.tscn").instantiate()
	get_tree().root.add_child(projectile)
	
	if attack_point:
		projectile.global_position = attack_point.global_position
	else:
		projectile.global_position = global_position + Vector3(0, 1.5, 0)
	
	if current_target:
		var direction = (current_target.global_position - projectile.global_position).normalized()
		projectile.set_direction(direction, projectile_speed, attack_damage, faction_id)

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
	elif anim_name == "attack_bow":
		current_animation_state = AnimationState.IDLE
