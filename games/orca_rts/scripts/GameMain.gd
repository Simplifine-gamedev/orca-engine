extends Node3D

## Main game script for Orca RTS
## Demonstrates the optimized wall building system

@onready var wall_system: WallSystem = $WallSystem
@onready var camera: Camera3D = $Camera3D
@onready var loading_indicator: WallLoadingIndicator = $UI/WallLoadingIndicator

var current_wall_type: String = "basic"
var is_in_build_mode: bool = false

func _ready():
	# Ensure wall system is set up
	if not wall_system:
		wall_system = WallSystem.new()
		wall_system.name = "WallSystem"
		add_child(wall_system)
	
	# Set up loading indicator
	if loading_indicator:
		loading_indicator.set_wall_system(wall_system)
	
	# Set up camera if it doesn't exist
	if not camera:
		camera = Camera3D.new()
		camera.name = "Camera3D"
		camera.position = Vector3(0, 10, 10)
		camera.look_at(Vector3.ZERO)
		add_child(camera)
	
	print("[GameMain] Game initialized")
	print("[GameMain] Press 'B' to enter wall build mode")
	print("[GameMain] Press 'ESC' to exit build mode")
	print("[GameMain] Press '1' for basic wall, '2' for reinforced wall")

func _input(event):
	# Toggle build mode
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_B:
			_toggle_build_mode()
		elif event.keycode == KEY_ESCAPE and is_in_build_mode:
			_exit_build_mode()
		elif event.keycode == KEY_1:
			current_wall_type = "basic"
			print("[GameMain] Selected basic wall")
		elif event.keycode == KEY_2:
			current_wall_type = "reinforced"
			print("[GameMain] Selected reinforced wall")
	
	# Place wall on mouse click
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT and is_in_build_mode:
			_place_wall_at_mouse()

func _process(_delta):
	if is_in_build_mode:
		_update_wall_preview_position()

func _toggle_build_mode():
	if is_in_build_mode:
		_exit_build_mode()
	else:
		_enter_build_mode()

func _enter_build_mode():
	if not wall_system.is_ready_for_build_mode():
		print("[GameMain] Wall system not ready yet, waiting for assets to load...")
		# The loading indicator will show automatically
		await wall_system.wall_preview_loaded
	
	wall_system.enter_build_mode(current_wall_type)
	is_in_build_mode = true
	print("[GameMain] Entered build mode - Move mouse to position wall, click to place")

func _exit_build_mode():
	wall_system.exit_build_mode()
	is_in_build_mode = false
	print("[GameMain] Exited build mode")

func _update_wall_preview_position():
	# Get mouse position in 3D world
	var mouse_pos = get_viewport().get_mouse_position()
	var from = camera.project_ray_origin(mouse_pos)
	var to = from + camera.project_ray_normal(mouse_pos) * 1000.0
	
	# Raycast to ground plane (y=0)
	var space_state = get_world_3d().direct_space_state
	var query = PhysicsRayQueryParameters3D.create(from, to)
	var result = space_state.intersect_ray(query)
	
	var target_pos: Vector3
	if result:
		target_pos = result.position
	else:
		# Project to ground plane if no hit
		target_pos = _project_to_ground_plane(from, to)
	
	# Snap to grid
	target_pos = _snap_to_grid(target_pos, 1.0)
	
	wall_system.update_preview_position(target_pos)

func _place_wall_at_mouse():
	var wall = wall_system.place_wall(current_wall_type)
	if wall:
		print("[GameMain] Wall placed successfully")

func _project_to_ground_plane(from: Vector3, to: Vector3) -> Vector3:
	# Project ray onto y=0 plane
	var direction = (to - from).normalized()
	var t = -from.y / direction.y
	return from + direction * t

func _snap_to_grid(pos: Vector3, grid_size: float) -> Vector3:
	return Vector3(
		round(pos.x / grid_size) * grid_size,
		round(pos.y / grid_size) * grid_size,
		round(pos.z / grid_size) * grid_size
	)
